## Developer Notes

* Do not update version by overwriting directly, to instead by globally replacing

* To convert argument type from `str` to `bool`, use `bool(strtobool(xxx))`

* We won't leverage `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` to create a new training job here
  since they reuse processes in the pool (i.e., a new job can have the same process ID with an old job),
  that is beneficial for resource usage but, in our case, causes GPU memory occupied by the old job.
  Instead, we use naive `Process` to do so, although the defunct (zombie) process is produced after which has finished (or exception occurred),
  it will get cleaned up in the next `process.start()` (or main process exited).

* If complicate types (e.g.: `str`, `int`, `float`, `bool`, `list`, `tuple`, ...) are provided for argument parser,
  it's better to use unified type `str` since some of them cannot be directly parsed by argument parser (e.g.: `bool`, `list`, `tuple`)

* In the training phase, do not switch to `eval` mode which causes some initializations of model invalid, such as
  `frozen backbone levels` and `frozen batch normalization modules`.

* We use custom type `REQUIRED` in `Config` instead of not assigning value for two reasons:

    1. The method of not assigning value required fields must be put in the head, but we want to keep their order in our expectations.
    
    2. The method of not assigning value does not work with the derived class.
