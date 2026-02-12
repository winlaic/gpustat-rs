# gpustat-rs

A Rust impelemtation of the tool [`gpustat`](https://github.com/wookayin/gpustat).

It doesn't need starting python or importing heavy libraries so it is faster to start.

## Speed Test Result

|Executable|On Remote Drive|On Local Drive|
|:-:|--:|--:|
|`gpustat-rs`|0.921s|0.171s|
|`gpustat`|1.521s|0.390s|
