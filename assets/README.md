# assets

Images and audio clips used by the demos.

## input.wav

Used by the first example in the README (`vm.ingest(audio="input.wav")`).
It says (in Mandarin) "I'm vegetarian and allergic to nuts." 5.8 seconds, 16 kHz mono PCM16,
synthesised with OpenAI TTS. It ends with 1 second of silence -- the streaming path needs 0.5 seconds
of continuous silence to decide the speaker has finished, so if the audio ended right at the last word,
`turn_over` would never arrive.

## question.wav

The clip fed into `vm.stream()` in the README's streaming section. It says (in Mandarin) "What are my
dietary restrictions?", 3.2 seconds, 16 kHz mono PCM16, synthesised with macOS `say -v Tingting`.

**It is a question** -- that is what the streaming demo shows: memory is looked up before the person has
finished speaking, and the final `ingest()` is judged not worth storing (`facts_count` 0) because the
sentence contains no new fact about the user. It also ends with just over 1 second of silence so that
`turn_over` arrives.

## speech.wav

Default input of `examples/02_streaming.py`.
It says (in Mandarin) "I like eating macarons", 7.7 seconds, 16 kHz mono PCM16.

## cafe_song.wav

"The song I heard at the cafe" -- the web demo plays this original clip back when asked about it.

15 seconds, 16 kHz mono PCM16. Mixed from two sources:

| Layer | Source | License |
|---|---|---|
| Music | *Chili Pepper* -- Fred Longshaw, 1927 jazz piano recording ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Chili_Pepper_by_Fred_Longshaw_(1927,_Jazz_piano).opus)) | Public domain (1927 recording) |
| Ambience | *Restaurant ambience* ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Restaurant_ambience.ogg)) | See the Commons file page |

Mix settings (music pushed into the background, as if coming from the shop's speakers):

```bash
ffmpeg -ss 8 -t 15 -i music.opus -stream_loop -1 -t 15 -i amb.ogg \
  -filter_complex "[0:a]volume=0.55,highpass=f=120,lowpass=f=6500[m];\
[1:a]volume=1.0[a];[m][a]amix=inputs=2:duration=first:normalize=0,\
dynaudnorm=p=0.7,alimiter=limit=0.95[out]" \
  -map "[out]" -ac 1 -ar 16000 -c:a pcm_s16le cafe_song.wav
```

To use your own recording: just overwrite this file (the archive table stores the path), but **re-running
ingest is recommended** -- the `tune:` / `scene:` tags are computed from the audio, so they will no longer
match once the content changes.
