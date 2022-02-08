## Troubleshooting

* Intel MKL Error

    * Error message
    
        ```
        Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
        ```

    * Solution
    
        ```
        $ export MKL_THREADING_LAYER=GNU
        ```
    
        or
        
        ```
        > import os
        > os.environ['MKL_THREADING_LAYER'] = 'GNU'
        ```

* RuntimeError: unable to open shared memory object

    * Solution

        ```
        $ ulimit -a | grep "open files"
        $ ulimit -SHn 51200
        $ ulimit -a | grep "open files"
        ```
