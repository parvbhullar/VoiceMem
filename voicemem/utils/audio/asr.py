"""语音转文字：流式识别（实时 partial）+ 非流式精转写（最终文本）。

流式两个实现，接口一致（``feed(samples) -> 累积文本`` / ``flush()`` / ``reset()``），
由 ``utils/defaults.py`` 的 ``asr`` 工厂按 ``VOICEMEM_ASR`` 选：

  · ``FunASRStreamingASR``  FunASR paraformer-zh-streaming（**默认**，中文更准）
  · ``StreamingASR``        sherpa-onnx 流式 zipformer（中英双语、纯 onnx 无 torch）
"""
from __future__ import annotations

import os
import re
import threading

import numpy as np

SAMPLE_RATE = 16000

SENSEVOICE_EMOTION_MAP = {
    "NEUTRAL": "中性",
    "HAPPY": "开心",
    "ANGRY": "愤怒",
    "SAD": "悲伤",
    "FEARFUL": "恐惧",
    "FEAR": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}


def pick_device() -> str:
    """自动选最佳设备: cuda > mps(Apple M) > cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class StreamingASR:
    """sherpa-onnx 流式 zipformer，出实时 partial 文本。``VOICEMEM_ASR=sherpa`` 时启用。"""

    def __init__(self, asr_dir: str) -> None:
        import sherpa_onnx          # 惰性：默认走 FunASR 时不拉 sherpa
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{asr_dir}/tokens.txt",
            encoder=f"{asr_dir}/encoder-epoch-99-avg-1.onnx",
            decoder=f"{asr_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{asr_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()

    def feed(self, samples):
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return self.rec.get_result(self.stream)

    def flush(self) -> str:
        """接口对齐 FunASRStreamingASR；sherpa 逐帧就把结果吐完了，没有尾巴要补。"""
        return self.rec.get_result(self.stream)

    def reset(self) -> None:
        self.stream = self.rec.create_stream()


# ── 默认流式 ASR：FunASR paraformer-zh-streaming ────────────────────────────────

class FunASRStreamingASR:
    """FunASR ``paraformer-zh-streaming``，出实时 partial 文本（核心默认流式 ASR）。

    paraformer 是**按块**推理的（``chunk_size=[0,10,5]`` → 600ms/块），而
    ``VoiceStream.feed()`` 喂进来的帧长由调用方决定（web 前端发 20ms 一帧），所以这里
    内部攒够 600ms 才推一次；``feed()`` 与 sherpa 版语义一致，返回**累积**文本
    （``VoiceStream`` 是 ``self._text = asr.feed(frame)`` 赋值语义，不能返回增量）。

    ``flush()``：VAD 判说完时由 ``VoiceStream`` 调一次——把不足一块的尾巴补零、跑一次
    ``is_final=True``，取出解码器 look-ahead 里还没吐的最后几个字。flush 之后若又来
    音频（说到一半停顿又续上），自动起一条新的子流继续累积，不会把这一轮的文本丢掉。
    """

    CHUNK_SIZE = [0, 10, 5]                    # paraformer-streaming 标配
    STRIDE     = CHUNK_SIZE[1] * 960           # 9600 samples @16k = 600ms
    LOOK_BACK  = dict(encoder_chunk_look_back=4, decoder_chunk_look_back=1)

    #: 离线包里的位置。跟回退那套并列放在 asr/ 下——两个都是流式 ASR，区别只是
    #: 默认(FunASR，中文更准) / 回退(sherpa，纯 onnx 不依赖 torch)。
    LOCAL_DIR = "funasr-paraformer-zh-streaming"

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        import logging as _logging
        import os as _os

        # `import funasr` 这一行本身就会把 **root logger** 从 WARNING 拉到 INFO 并挂
        # 一个 handler（实测：import 前 WARNING/0 handlers，import 后 INFO/1）。
        # 后果不只是它自己刷屏——之后 openai/httpx 的 INFO 也全冒出来，一次基础用法
        # 能刷几十行 "HTTP Request: POST ... 200 OK"，真正的结果被埋在中间。
        # 记下 import 前的状态，import 完原样恢复。VOICEMEM_VERBOSE=1 保留原样。
        _quiet = _os.environ.get("VOICEMEM_VERBOSE", "0") == "0"
        _root = _logging.getLogger()
        _lvl, _handlers = _root.level, list(_root.handlers)

        from funasr import AutoModel          # 惰性：只有真用流式 ASR 才拉 funasr

        if _quiet:
            _os.environ.setdefault("TQDM_DISABLE", "1")   # 每转写一块刷一条 rtf 进度条
            _root.setLevel(_lvl)
            for _h in list(_root.handlers):
                if _h not in _handlers:
                    _root.removeHandler(_h)
        # funasr 的 AutoModel 里有一句 logging.basicConfig(level=log_level)，默认
        # INFO —— 它设的是 **root**，于是 openai/httpx 那些库的 INFO 也跟着全冒出来
        # （"HTTP Request: POST ... 200 OK" 刷几十行）。在 voicemem/__init__ 里给
        # 各个 logger 设等级挡不住这个，因为它改的是 root。直接把参数传进去。
        if model is None:
            # 有离线包就用本地，没有就交给 funasr 按模型名自己下（848M，会卡在
            # 用户说的第一句上——所以离线包里带着它，别人拉下来开箱即用）。
            from voicemem.utils.common.paths import models_dir
            local = models_dir() / "asr" / self.LOCAL_DIR
            model = str(local) if (local / "config.yaml").exists() else "paraformer-zh-streaming"
        self.model = AutoModel(model=model, device=device or pick_device(),
                               disable_update=True,
                               log_level="ERROR" if _quiet else "INFO")
        self.reset()

    def _run(self, samples, is_final: bool) -> str:
        res = self.model.generate(input=samples, cache=self._cache, is_final=is_final,
                                  chunk_size=self.CHUNK_SIZE, **self.LOOK_BACK)
        if res and res[0].get("text"):
            self._text += res[0]["text"]
        return self._text

    def feed(self, samples) -> str:
        """喂任意长度的 16k float32 帧；攒够 600ms 推一块。返回累积文本。"""
        if self._final:                        # 上一轮 flush 过又来音频 → 起新子流续着攒
            self._cache, self._final = {}, False
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        while len(self._buf) >= self.STRIDE:
            self._run(self._buf[:self.STRIDE], False)
            self._buf = self._buf[self.STRIDE:]
        return self._text

    def flush(self) -> str:
        """VAD 判说完时调：尾巴补零跑 is_final=True，别丢最后几个字。幂等。"""
        if self._final:
            return self._text
        tail = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        tail = (np.pad(tail, (0, self.STRIDE - len(tail))) if len(tail)
                else np.zeros(self.STRIDE, dtype=np.float32))
        self._final = True
        return self._run(tail, True)

    def reset(self) -> None:
        self._cache: dict = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._final = False


class Transcriber:
    """SenseVoiceSmall 出最终文本（中英），比流式 ASR 更准，锁定一轮时用这个。

    ``language`` 默认 ``auto``（也可用 ``VOICEMEM_ASR_LANGUAGE`` 配）。之前写死
    ``zh``：SenseVoice 是多语模型，被钉在中文上时英文语音会被硬塞成汉字——
    实测说 "hello how are you" 转出来是「你不知道还lohow areyou」。转错的文本
    接着进抽取和检索，后面一路都是垃圾。SenseVoice 认的值：auto / zh / en /
    yue / ja / ko / nospeech。"""

    def __init__(self, device: str, language: str = "") -> None:
        self.language = (language
                         or os.environ.get("VOICEMEM_ASR_LANGUAGE", "")
                         or "auto")
        from funasr import AutoModel        # 懒 import：只有用非流式精转写才需要 funasr
        from voicemem.utils.common.paths import hf_model
        _name = hf_model("emotion", "FunAudioLLM/SenseVoiceSmall", "asr")
        self.model = AutoModel(model=_name, hub="hf",
                               device=device, disable_update=True,
                               trust_remote_code=False)

    def _generate(self, audio) -> str:
        res = self.model.generate(input=audio, cache={}, language=self.language,
                                  use_itn=True, ban_emo_unk=True)
        if not res:
            return ""
        return res[0].get("text", "") or ""

    def run(self, audio) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._generate(audio)).strip()

    def run_with_emotion(self, audio) -> tuple[str, str]:
        """一次 SenseVoice 推理同时取得文本和声学情绪 token。"""
        raw = self._generate(audio)
        tags = re.findall(r"<\|([^|]+)\|>", raw.upper())
        emotion = next((SENSEVOICE_EMOTION_MAP[tag] for tag in tags
                        if tag in SENSEVOICE_EMOTION_MAP), "中性")
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emotion


# ── API 流式 ASR：不下模型，走 OpenAI 兼容的转写接口 ────────────────────────────

def _wav_bytes(samples, rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1,1] → 内存里的 16-bit WAV。转写接口要的是文件，不是裸 PCM。"""
    import io
    import wave
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class OpenAIStreamingASR:
    """转写走 API，本地一个模型都不下。``VOICEMEM_ASR=openai`` 时启用。

    为什么不是"攒完一轮再转一次"：``stream.py`` 判「说完了」的条件里有
    ``self._text.strip()``——``feed()`` 一直返回空串的话这一轮永远不会结束，
    ``flush()`` 也就永远不会被调用。而且投机预取吃的就是 partial 文本，没有
    partial 就等于把这个 demo 的核心（说到一半就把记忆搜好）关掉了。

    所以 partial 也走网络，但**同时只允许一个请求在飞**（single-flight）：
    ``feed()`` 立刻返回上一次已经拿到的文本，新请求在后台线程里跑，回来了下一次
    ``feed()`` 就能看到。说话人还在说的时候，这条路是完全非阻塞的。

    ``flush()`` 是唯一一次同步等待：那一次的结果就是这一轮的最终文本。

    代价说清楚：每 ``partial_every_s`` 秒一次网络请求（默认 0.9s，单飞所以慢的
    时候自动降频），加收尾一次。多语言、自动判语种，中文/英文/印地语都能转——
    这正是本地那两个模型做不到的地方。
    """

    def __init__(self, model: str = "", language: str = "",
                 transcribe=None, partial_every_s: float = 0.9) -> None:
        self.model = model or os.environ.get("VOICEMEM_ASR_MODEL", "") or "gpt-4o-mini-transcribe"
        lang = (language or os.environ.get("VOICEMEM_ASR_LANGUAGE", "") or "auto").lower()
        self.language = "" if lang in ("", "auto") else lang
        self.partial_every_s = partial_every_s
        self._transcribe = transcribe or self._api_transcribe
        self._client = None
        self._lock = threading.Lock()
        self.reset()

    # ── 网络那一半 ───────────────────────────────────────────────────────────
    def _api_transcribe(self, wav: bytes) -> str:
        if self._client is None:
            from openai import OpenAI
            from voicemem.llm_config import resolve_api_key, resolve_base_url
            self._client = OpenAI(api_key=resolve_api_key(None),
                                  base_url=resolve_base_url(None))
        kw = {"language": self.language} if self.language else {}
        r = self._client.audio.transcriptions.create(
            model=self.model, file=("turn.wav", wav, "audio/wav"), **kw)
        return (getattr(r, "text", "") or "").strip()

    def _run(self, audio) -> None:
        """后台线程：转写当前缓冲，成功了就更新 _text。失败不抛——一次 partial
        丢了没关系，flush() 还会再转一次完整的。"""
        try:
            text = self._transcribe(_wav_bytes(audio))
        except Exception as e:                       # noqa: BLE001
            print(f"[asr] partial 转写失败（忽略）：{type(e).__name__}: {e}", flush=True)
            return
        with self._lock:
            if text:
                self._text = text

    # ── 流式接口（跟另外两个 ASR 一致）────────────────────────────────────────
    def feed(self, samples) -> str:
        self._buf.append(np.asarray(samples, dtype=np.float32))
        self._n += len(samples)
        if self._inflight is None or not self._inflight.is_alive():
            since = (self._n - self._asked_at) / SAMPLE_RATE
            if since >= self.partial_every_s:
                self._asked_at = self._n
                audio = np.concatenate(self._buf)
                self._inflight = threading.Thread(target=self._run, args=(audio,), daemon=True)
                self._inflight.start()
        with self._lock:
            return self._text

    def flush(self) -> str:
        """这一轮的最终文本。唯一一次同步等待。"""
        if not self._buf:
            return self._text
        audio = np.concatenate(self._buf)
        if len(audio) < int(0.2 * SAMPLE_RATE):      # 太短，转了也是噪声
            return self._text
        try:
            text = self._transcribe(_wav_bytes(audio))
        except Exception as e:                       # noqa: BLE001
            print(f"[asr] 收尾转写失败：{type(e).__name__}: {e}", flush=True)
            return self._text
        if text:
            with self._lock:
                self._text = text
        return self._text

    def warmup(self) -> None:
        """把第一次网络调用的开销挪到启动时。

        实测：进程里第一次转写 ~2.0s，之后每次 ~0.65s。差的那一秒半全落在
        用户说的**第一句**上——而第一句恰好决定了这个 demo 给人的快慢印象。
        DNS、TLS、client 构造都在这一秒半里，发一段静音就能把它们付掉。

        失败不抛：预热是优化，不该让服务起不来（开机时没网也照样能起）。
        而且就算服务端拒了这段静音，TLS 和连接也已经建好了，目的照样达成。
        """
        try:
            self._transcribe(_wav_bytes(np.zeros(int(0.3 * SAMPLE_RATE),
                                                dtype=np.float32)))
        except Exception as e:  # noqa: BLE001
            print(f"[asr] 预热失败（忽略）：{type(e).__name__}: {e}", flush=True)
        finally:
            # 预热的结果绝不能留在 _text 里，否则第一轮开口前就已经有文本了,
            # stream.py 会拿它当成用户说了话。
            self.reset()

    def reset(self) -> None:
        self._buf: list = []
        self._n = 0
        self._asked_at = 0
        self._inflight = None
        self._text = ""
