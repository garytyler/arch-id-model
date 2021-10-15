# Development

### Close a running tensorboard process

`kill $(ps -e | grep 'tensorboard' | awk '{print $1}')`

### Tensorflow logging levels

See: https://stackoverflow.com/a/40982782

Control tensorflow logging output with environment variable `TF_CPP_MIN_LOG_LEVEL`.

- 0 = all messages are logged (default behavior)
- 1 = `INFO` messages are not printed
- 2 = `INFO` and `WARNING` messages are not printed
- 3 = `INFO`, `WARNING`, and `ERROR` messages are not printed
