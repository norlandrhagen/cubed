Search.setIndex({"docnames": ["api", "array-api", "computation", "contributing", "design", "generated/cubed.Array", "generated/cubed.Array.compute", "generated/cubed.Array.rechunk", "generated/cubed.Array.visualize", "generated/cubed.Callback", "generated/cubed.Spec", "generated/cubed.TaskEndEvent", "generated/cubed.apply_gufunc", "generated/cubed.compute", "generated/cubed.from_array", "generated/cubed.from_zarr", "generated/cubed.map_blocks", "generated/cubed.measure_reserved_mem", "generated/cubed.nanmean", "generated/cubed.nansum", "generated/cubed.random.random", "generated/cubed.runtime.executors.beam.BeamDagExecutor", "generated/cubed.runtime.executors.lithops.LithopsDagExecutor", "generated/cubed.runtime.executors.modal_async.AsyncModalDagExecutor", "generated/cubed.runtime.executors.python.PythonDagExecutor", "generated/cubed.store", "generated/cubed.to_zarr", "generated/cubed.visualize", "getting-started/demo", "getting-started/index", "getting-started/installation", "getting-started/why-cubed", "index", "operations", "related-projects", "user-guide/executors", "user-guide/index", "user-guide/memory", "user-guide/reliability", "user-guide/scaling", "user-guide/storage"], "filenames": ["api.rst", "array-api.md", "computation.md", "contributing.md", "design.md", "generated/cubed.Array.rst", "generated/cubed.Array.compute.rst", "generated/cubed.Array.rechunk.rst", "generated/cubed.Array.visualize.rst", "generated/cubed.Callback.rst", "generated/cubed.Spec.rst", "generated/cubed.TaskEndEvent.rst", "generated/cubed.apply_gufunc.rst", "generated/cubed.compute.rst", "generated/cubed.from_array.rst", "generated/cubed.from_zarr.rst", "generated/cubed.map_blocks.rst", "generated/cubed.measure_reserved_mem.rst", "generated/cubed.nanmean.rst", "generated/cubed.nansum.rst", "generated/cubed.random.random.rst", "generated/cubed.runtime.executors.beam.BeamDagExecutor.rst", "generated/cubed.runtime.executors.lithops.LithopsDagExecutor.rst", "generated/cubed.runtime.executors.modal_async.AsyncModalDagExecutor.rst", "generated/cubed.runtime.executors.python.PythonDagExecutor.rst", "generated/cubed.store.rst", "generated/cubed.to_zarr.rst", "generated/cubed.visualize.rst", "getting-started/demo.md", "getting-started/index.md", "getting-started/installation.md", "getting-started/why-cubed.md", "index.md", "operations.md", "related-projects.md", "user-guide/executors.md", "user-guide/index.md", "user-guide/memory.md", "user-guide/reliability.md", "user-guide/scaling.md", "user-guide/storage.md"], "titles": ["API Reference", "Python Array API", "Computation", "Contributing", "Design", "cubed.Array", "cubed.Array.compute", "cubed.Array.rechunk", "cubed.Array.visualize", "cubed.Callback", "cubed.Spec", "cubed.TaskEndEvent", "cubed.apply_gufunc", "cubed.compute", "cubed.from_array", "cubed.from_zarr", "cubed.map_blocks", "cubed.measure_reserved_mem", "cubed.nanmean", "cubed.nansum", "cubed.random.random", "cubed.runtime.executors.beam.BeamDagExecutor", "cubed.runtime.executors.lithops.LithopsDagExecutor", "cubed.runtime.executors.modal_async.AsyncModalDagExecutor", "cubed.runtime.executors.python.PythonDagExecutor", "cubed.store", "cubed.to_zarr", "cubed.visualize", "Demo", "Getting Started", "Installation", "Why Cubed?", "Cubed", "Operations", "Related Projects", "Executors", "User Guide", "Memory", "Reliability", "Scaling", "Storage"], "terms": {"A": [0, 2, 31, 37, 38, 39], "cube": [0, 2, 3, 4, 28, 29, 30, 33, 34, 35, 36, 37, 38, 40], "can": [0, 2, 8, 10, 17, 27, 30, 31, 34, 35, 37, 38, 39, 40], "creat": [0, 3, 14, 17, 33, 34, 35, 40], "from_arrai": 0, "from_zarr": 0, "one": [0, 2, 31, 33, 38, 39, 40], "python": [0, 3, 4, 5, 12, 17, 25, 26, 28, 30, 32, 34, 36, 37], "creation": [0, 1], "implement": [1, 4, 17, 25, 28, 31, 32, 33, 34, 39], "array_api": [1, 28], "refer": [1, 12, 32, 35, 39], "its": [1, 4, 7, 33, 37, 39], "specif": [1, 10, 32, 39], "document": 1, "The": [1, 2, 4, 6, 7, 8, 10, 13, 15, 17, 24, 25, 26, 27, 33, 34, 35, 37, 39, 40], "follow": [1, 4, 32, 33, 37, 40], "part": [1, 33, 35, 39], "ar": [1, 2, 3, 4, 8, 12, 27, 30, 31, 33, 34, 35, 37, 38, 39, 40], "categori": 1, "object": [1, 2, 9, 14, 17, 25, 28, 40], "function": [1, 2, 4, 6, 8, 12, 13, 16, 17, 27, 30, 32, 33, 35, 37, 39], "In": [1, 25, 33, 34, 37, 39, 40], "place": 1, "op": 1, "from_dlpack": 1, "index": [1, 33], "boolean": 1, "manipul": 1, "flip": 1, "roll": 1, "search": 1, "nonzero": 1, "set": [1, 2, 6, 10, 12, 13, 17, 30, 37, 38, 39, 40], "unique_al": 1, "unique_count": 1, "unique_invers": 1, "unique_valu": 1, "sort": 1, "argsort": 1, "statist": 1, "std": 1, "var": 1, "accept": 1, "extra": 1, "chunk": [1, 2, 4, 5, 7, 14, 16, 20, 28, 31, 32, 33, 34, 35, 36, 38, 39], "spec": [1, 5, 6, 13, 14, 15, 16, 17, 20, 28, 35, 37, 40], "keyword": [1, 17], "argument": [1, 17], "arang": 1, "start": [1, 28, 31, 32, 35, 39], "stop": 1, "none": [1, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27], "step": 1, "1": [1, 11, 20, 28, 40], "dtype": [1, 2, 4, 16, 19, 33, 37], "devic": 1, "auto": [1, 14], "asarrai": [1, 14, 28], "obj": 1, "copi": 1, "empti": [1, 33], "shape": [1, 4, 7, 33], "empty_lik": 1, "x": [1, 14, 18, 19, 25, 26], "ey": 1, "n_row": 1, "n_col": 1, "k": 1, "0": [1, 10, 20, 30], "full": [1, 30, 39], "fill_valu": 1, "full_lik": 1, "linspac": 1, "num": 1, "endpoint": 1, "true": [1, 6, 8, 13, 27, 39], "ones": [1, 34], "ones_lik": 1, "zero": [1, 19], "zeros_lik": 1, "broadcast_to": 1, "ha": [2, 10, 28, 30, 31, 33, 34, 35, 37, 38, 39], "lazi": [2, 28], "model": [2, 4, 31], "As": [2, 37], "arrai": [2, 9, 12, 13, 14, 15, 16, 19, 25, 26, 27, 28, 31, 33, 34, 35, 37, 39, 40], "invok": 2, "i": [2, 4, 6, 8, 12, 13, 17, 25, 26, 27, 28, 31, 32, 33, 34, 36, 37, 38, 39, 40], "built": [2, 4], "up": [2, 35, 37, 38, 39, 40], "onli": [2, 6, 8, 13, 27, 31, 33, 34, 35, 38, 39], "when": [2, 10, 17, 34, 35, 37, 38, 39, 40], "explicitli": 2, "trigger": 2, "call": [2, 33, 34, 35, 38], "implicitli": 2, "convert": [2, 34, 39], "an": [2, 3, 7, 8, 10, 14, 15, 17, 21, 22, 23, 26, 27, 31, 33, 36, 37, 38, 39], "numpi": [2, 4, 28, 34], "disk": [2, 4, 8, 27], "zarr": [2, 4, 5, 15, 25, 26, 31, 32, 35, 37, 38, 39, 40], "represent": [2, 39], "direct": 2, "acycl": 2, "graph": [2, 6, 8, 13, 27, 34], "dag": 2, "where": [2, 10, 34, 39], "node": [2, 31], "edg": 2, "express": 2, "primit": [2, 32, 33, 34], "oper": [2, 8, 17, 25, 26, 27, 31, 32, 34, 37, 38, 39, 40], "For": [2, 37, 38, 39, 40], "exampl": [2, 4, 28, 32, 33, 35, 37, 38, 39], "mai": [2, 4, 33, 35, 39, 40], "rechunk": [2, 4, 31, 32, 34, 39], "anoth": [2, 31, 37, 38, 39], "us": [2, 4, 6, 8, 10, 13, 15, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40], "Or": 2, "pair": 2, "ad": [2, 39], "togeth": [2, 39], "blockwis": [2, 32, 34, 39], "both": [2, 34, 38, 39], "have": [2, 6, 13, 34, 35, 37, 38, 39], "requir": [2, 3, 17, 31, 35, 37, 39], "known": [2, 31, 37, 39], "ahead": [2, 37], "time": [2, 31, 33, 37, 38, 39, 40], "each": [2, 33, 37, 38, 39], "run": [2, 6, 10, 13, 17, 24, 25, 26, 28, 30, 31, 34, 35, 37, 38, 40], "task": [2, 4, 10, 11, 12, 17, 24, 34, 35, 37, 38, 39], "output": [2, 8, 12, 26, 27, 33, 37, 38, 39], "need": [2, 4, 12, 33, 37, 39, 40], "size": [2, 20, 31, 33, 36, 39], "natur": [2, 34, 37], "which": [2, 4, 8, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40], "while": [2, 6, 13, 33, 37, 39], "build": [2, 35, 37], "see": [2, 28, 30, 39], "discuss": [2, 35, 37], "travers": 2, "materi": 2, "write": [2, 8, 25, 27, 31, 37, 38, 39, 40], "them": [2, 33, 38, 40], "storag": [2, 5, 10, 15, 26, 31, 32, 36, 38, 39], "detail": [2, 33, 35], "how": [2, 31, 33, 37, 39, 40], "depend": [2, 6, 29, 32, 38, 39], "runtim": [2, 17, 25, 26, 31, 32, 34, 35, 37, 40], "distribut": [2, 30, 31, 34, 38, 39], "choos": 2, "don": [2, 35, 40], "t": [2, 6, 8, 13, 27, 34, 35, 37, 40], "parallel": [2, 31, 34, 39], "effici": [2, 37], "thi": [2, 4, 6, 7, 8, 10, 12, 17, 25, 26, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40], "process": [2, 4, 17, 25, 26, 31, 34, 35, 37, 38, 39], "advantag": [2, 34, 39], "disadvantag": [2, 39], "One": [2, 39], "sinc": [2, 31, 33, 34, 35, 37, 38], "shuffl": [2, 31, 39], "involv": [2, 3, 31, 39], "straightforward": [2, 35], "scale": [2, 31, 32, 36, 38], "veri": [2, 3, 31, 35, 39], "high": [2, 31, 34, 37], "level": [2, 31, 34], "serverless": [2, 31, 34, 35, 39], "environ": [2, 3, 35, 39], "also": [2, 28, 30, 35, 38, 39], "make": [2, 34, 37, 39], "multipl": [2, 12, 13, 16, 27, 32, 33, 39], "engin": [2, 21, 22, 23, 24, 31], "main": 2, "everi": [2, 4, 35, 39], "intermedi": [2, 6, 10, 13, 17, 36, 39], "written": [2, 38], "slow": [2, 38], "howev": [2, 12, 31, 39], "opportun": 2, "optim": [2, 6, 8, 13, 27, 31], "befor": [2, 6, 8, 13, 27, 35, 39, 40], "map": [2, 6, 8, 13, 27, 33, 39], "fusion": [2, 6, 8, 13, 27], "welcom": 3, "pleas": 3, "head": 3, "over": [3, 19, 33, 40], "github": 3, "get": [3, 32, 35, 36, 39, 40], "conda": [3, 29], "name": [3, 5, 8, 27, 34, 40], "3": [3, 28], "9": [3, 28], "activ": 3, "pip": [3, 29, 35], "instal": [3, 17, 29, 32], "r": 3, "txt": [3, 35], "e": [3, 39], "compos": [4, 33], "five": 4, "layer": [4, 34], "from": [4, 6, 13, 14, 15, 16, 31, 32, 33, 35, 37, 38, 39], "bottom": [4, 33], "top": [4, 33], "blue": 4, "block": [4, 16, 33, 39], "green": [4, 33], "red": 4, "other": [4, 36], "project": [4, 30, 32, 36], "like": [4, 14, 25, 28, 31, 33, 34, 37, 40], "beam": [4, 30, 31, 32, 34, 39], "let": 4, "": [4, 6, 23, 31, 32, 33, 34, 37, 39, 40], "go": [4, 31], "through": 4, "back": [4, 5, 35], "mean": [4, 18, 37, 38, 39], "type": [4, 7, 8, 15, 17, 25, 26, 27, 36], "inherit": 4, "attribut": [4, 5, 10, 11, 33], "includ": [4, 8, 10, 27, 34], "underli": 4, "store": [4, 10, 15, 17, 26, 39, 40], "local": [4, 28, 30, 36, 37, 40], "cloud": [4, 17, 28, 31, 34, 36, 37, 38], "well": [4, 34, 37, 39], "unit": [4, 10], "comput": [4, 8, 9, 10, 15, 17, 18, 25, 26, 27, 28, 31, 32, 34, 35, 37, 38, 39, 40], "system": [4, 17, 31, 38, 39, 40], "extern": 4, "It": [4, 17, 33, 34, 35], "algorithm": [4, 33, 37], "deleg": 4, "stateless": [4, 31], "executor": [4, 6, 10, 13, 17, 25, 26, 28, 30, 32, 36, 37, 38, 40], "lithop": [4, 17, 30, 31, 32, 35], "modal": [4, 17, 23, 30, 32, 35, 39], "apach": [4, 21, 30, 31, 32], "There": [4, 37, 39], "two": [4, 33, 35, 39], "These": 4, "provid": [4, 8, 27, 31, 35, 37, 38], "all": [4, 31, 33, 35, 39], "new": [4, 12, 34, 37, 39], "wa": [4, 34, 39], "chosen": 4, "public": 4, "defin": [4, 34, 39], "subset": [4, 28], "few": [4, 31, 38, 39], "extens": [4, 8, 27], "io": [4, 32, 38], "random": [4, 32], "number": [4, 32, 33, 38, 39], "gener": [4, 12, 31, 32, 33, 37, 39], "map_block": [4, 32], "heavili": [4, 34], "dask": [4, 12, 28, 30, 31, 32, 39], "applic": 4, "class": [5, 9, 10, 11, 21, 22, 23, 24], "zarrai": 5, "plan": [5, 32, 34, 37], "conform": 5, "api": [5, 23, 28, 32, 33, 34], "standard": [5, 28, 32, 34], "__init__": [5, 9, 10, 11, 21, 22, 23, 24], "method": [5, 9, 10, 11, 21, 22, 23, 24], "callback": [6, 11, 13], "optimize_graph": [6, 8, 13, 27], "optimize_funct": [6, 8, 13, 27], "resum": [6, 13], "kwarg": [6, 12, 13, 16, 17, 22, 23, 25, 26], "ani": [6, 10, 17, 39, 40], "paramet": [6, 7, 8, 10, 12, 13, 15, 17, 25, 26, 27, 39], "option": [6, 8, 10, 13, 15, 17, 25, 26, 27, 29], "If": [6, 8, 10, 13, 27, 35, 37, 38, 40], "list": [6, 13, 30], "send": [6, 13], "event": [6, 9, 13], "bool": [6, 8, 13, 27], "otherwis": [6, 8, 13, 27], "default": [6, 8, 10, 13, 24, 25, 26, 27, 28, 38, 39, 40], "callabl": [6, 8, 13, 27], "perform": [6, 8, 13, 27, 31, 36, 37], "alreadi": [6, 13], "been": [6, 10, 13, 28, 31, 34, 35, 38, 39], "won": [6, 13], "recomput": [6, 13], "fals": [6, 8, 13, 18, 19, 27], "chang": [7, 33, 34, 39], "without": [7, 33, 35, 39], "data": [7, 10, 17, 31, 35, 36, 37], "tupl": 7, "desir": 7, "after": [7, 38, 40], "return": [7, 8, 15, 17, 19, 20, 27, 33], "corearrai": [7, 13, 27], "filenam": [8, 27], "format": [8, 27], "show_hidden": [8, 27], "produc": [8, 27, 37], "str": [8, 10, 17, 27], "file": [8, 27, 38, 40], "doesn": [8, 27, 37], "svg": [8, 27], "png": [8, 27], "pdf": [8, 27], "dot": [8, 27], "jpeg": [8, 27], "jpg": [8, 27], "render": [8, 27], "displai": [8, 27, 39], "show": [8, 27, 33, 35], "mark": [8, 27], "hidden": [8, 27], "ipython": [8, 27], "imag": [8, 27], "import": [8, 27, 28, 35, 37, 39, 40], "notebook": [8, 27], "receiv": 9, "dure": [9, 37], "work_dir": [10, 17, 28, 35, 40], "allowed_mem": [10, 28, 35, 37, 39], "reserved_mem": [10, 17, 37], "storage_opt": 10, "resourc": [10, 28, 39], "avail": [10, 28, 37, 38], "specifi": [10, 17, 18, 28, 36, 40], "directori": [10, 17, 40], "path": [10, 15, 17, 26], "fsspec": [10, 17, 40], "url": [10, 17, 40], "int": [10, 17], "total": [10, 38, 39], "memori": [10, 12, 17, 31, 34, 36, 39], "worker": [10, 31, 35, 37, 39], "byte": [10, 12, 17], "should": [10, 36, 37, 39, 40], "form": [10, 40], "valu": [10, 33, 37], "kb": 10, "mb": 10, "gb": 10, "tb": 10, "etc": [10, 32], "reserv": [10, 17, 36], "non": [10, 32, 37, 39], "dict": 10, "pass": [10, 17, 31, 33], "array_nam": 11, "num_task": 11, "task_create_tstamp": 11, "function_start_tstamp": 11, "function_end_tstamp": 11, "task_result_tstamp": 11, "peak_measured_mem_start": 11, "peak_measured_mem_end": 11, "inform": [11, 12, 37], "about": [11, 12, 34], "complet": [11, 30, 38, 39], "func": [12, 16], "signatur": 12, "arg": [12, 16], "ax": [12, 33], "axi": [12, 18, 19, 33], "output_dtyp": 12, "output_s": 12, "vector": 12, "appli": [12, 16, 39], "ufunc": 12, "similar": [12, 34], "cutdown": 12, "version": [12, 17], "equival": 12, "usag": [12, 31, 32, 37, 39], "current": [12, 25, 38, 39], "limit": [12, 39], "keepdim": [12, 18, 19], "allow_rechunk": 12, "support": [12, 33, 34, 35], "assum": 12, "alloc": [12, 31], "more": [12, 33, 34, 35, 37, 38, 39], "than": [12, 33, 34, 37, 38, 39, 40], "you": [12, 29, 30, 34, 35, 36, 37, 38, 39, 40], "tell": 12, "extra_projected_mem": 12, "amount": [12, 17, 31, 35, 37, 38, 39], "per": [12, 39], "onc": [13, 38], "load": 15, "string": [15, 26], "input": [15, 16, 33, 37, 39], "drop_axi": 16, "new_axi": 16, "correspond": [16, 33, 34], "measur": [17, 37, 39], "given": [17, 19, 37, 39], "exclud": 17, "vari": [17, 39], "packag": [17, 30, 34], "guid": [17, 29, 32], "work": [17, 31, 32, 33, 35, 37, 38, 40], "trivial": [17, 39], "tini": 17, "peak": 17, "must": [17, 25, 38], "report": [17, 37], "arithmet": 18, "along": [18, 33], "ignor": 18, "nan": [18, 19], "sum": [19, 37], "element": [19, 33, 39], "treat": 19, "float": 20, "half": 20, "open": 20, "interv": 20, "execut": [21, 22, 23, 24, 32, 37, 39], "async": 23, "sequenti": 24, "loop": 24, "sourc": [25, 39], "target": [25, 31], "save": [25, 26], "note": [25, 26, 33, 35, 38, 40], "eager": [25, 26], "immedi": [25, 26, 38], "collect": 25, "we": [28, 33, 37, 39], "ll": 28, "simpl": [28, 35], "xp": 28, "tmp": 28, "100kb": 28, "2": [28, 30], "4": 28, "5": 28, "6": 28, "7": [28, 30], "8": 28, "essenti": [28, 39], "convent": 28, "notic": 28, "just": [28, 33], "describ": [28, 35, 38], "b": 28, "c": [28, 30], "add": [28, 33, 39, 40], "evalu": 28, "so": [28, 34, 35, 37, 38, 39, 40], "noth": [28, 39], "yet": [28, 38, 39], "print": 28, "result": 28, "interact": 28, "10": 28, "readm": 28, "servic": [28, 31, 36, 37, 39, 40], "aim": [29, 37, 39], "quickli": [29, 38], "possibl": [29, 31, 37, 39], "why": [29, 32], "demo": [29, 32], "minim": 30, "forg": 30, "m": 30, "mani": [30, 33, 35], "differ": [30, 32, 33, 38], "especi": [30, 39], "diagnost": 30, "To": [30, 38, 39, 40], "optional_depend": 30, "pyproject": 30, "toml": 30, "tqdm": 30, "graphviz": 30, "jinja2": 30, "pydot": 30, "panda": 30, "matplotlib": 30, "seaborn": 30, "gcsf": 30, "aw": [30, 35, 39, 40], "client": [30, 38], "s3f": 30, "gcp": [30, 35], "coil": 30, "test": [30, 35, 38], "runner": [30, 34], "separ": [30, 39, 40], "due": 30, "conflict": 30, "req": 30, "dill": 30, "pytest": 30, "cov": 30, "mock": 30, "manag": [31, 37], "major": 31, "challeng": 31, "design": [31, 32, 39], "framework": 31, "hadoop": 31, "mapreduc": 31, "spark": 31, "purpos": 31, "lead": [31, 39], "widespread": 31, "adopt": 31, "success": 31, "user": [31, 34], "carefulli": 31, "configur": [31, 37, 38, 39], "understand": [31, 34, 39], "break": 31, "program": 31, "abstract": [31, 34], "disproportion": [31, 38, 39], "often": 31, "spent": [31, 39], "tune": [31, 39], "larg": [31, 37, 38, 39], "common": [31, 37], "theme": 31, "here": [31, 33, 39], "most": [31, 35, 36], "interest": 31, "embarrassingli": 31, "between": [31, 32, 33], "lot": [31, 40], "effort": 31, "put": [31, 37], "googl": [31, 34, 35, 38, 39, 40], "dataflow": [31, 34, 35], "lesser": 31, "extent": 31, "undoubtedli": 31, "improv": [31, 37, 39], "made": 31, "problem": [31, 39], "awai": 31, "approach": [31, 40], "gain": 31, "traction": 31, "last": 31, "year": [31, 34], "formerli": 31, "pywren": 31, "eschew": 31, "central": 31, "do": 31, "everyth": 31, "via": [31, 32], "case": [31, 37, 39, 40], "persist": [31, 39, 40], "n": 31, "dimension": 31, "guarante": [31, 32, 39], "even": [31, 38, 39], "though": [31, 39], "deliber": 31, "avoid": [31, 33], "instead": 31, "bulk": 31, "read": [31, 33, 34, 37, 38, 39], "alwai": 31, "tightli": [31, 37], "control": 31, "therebi": [31, 39], "unpredict": 31, "attempt": [31, 38], "further": [31, 37, 39], "bound": [31, 34, 37, 39], "librari": [32, 34], "maximum": [32, 37, 39], "integr": [32, 34], "xarrai": 32, "reliabl": [32, 35, 36], "standardis": 32, "miss": 32, "relat": 32, "previou": [32, 38], "core": [32, 33, 34], "tree": 32, "elemwis": 32, "map_direct": 32, "reduct": [32, 37, 39], "arg_reduct": 32, "contribut": 32, "look": 33, "depth": 33, "diagram": 33, "shown": 33, "white": 33, "middl": 33, "orang": 33, "pink": 33, "Not": 33, "repres": [33, 39], "select": [33, 40], "fundament": [33, 34], "simplest": [33, 39], "preserv": 33, "numblock": 33, "singl": [33, 34, 35, 38], "arrow": 33, "order": [33, 37, 39], "clutter": 33, "broadcast": 33, "thei": [33, 34, 37, 39], "match": [33, 39], "too": 33, "squeez": 33, "although": [33, 35], "second": [33, 35], "dimens": 33, "drop": 33, "allow": [33, 36, 38, 39], "directli": 33, "regard": 33, "boundari": 33, "No": 33, "turn": [33, 39], "same": [33, 39, 40], "structur": 33, "side": 33, "access": 33, "whatev": [33, 37], "wai": [33, 38, 39], "concat": 33, "sent": 33, "outer": 33, "three": [33, 34, 38], "consult": 33, "page": [33, 39], "reduc": [33, 39], "repeat": 33, "first": [33, 35, 37, 39], "round": 33, "combin": 33, "would": [33, 37, 39, 40], "until": 33, "similarli": 33, "rather": [33, 40], "flexibl": 34, "sever": 34, "compon": 34, "datafram": 34, "bag": 34, "delai": [34, 39], "decompos": 34, "fine": 34, "grain": 34, "higher": 34, "easier": 34, "visual": 34, "reason": [34, 37, 39], "newer": 34, "wherea": [34, 39], "varieti": [34, 35], "matur": [34, 35], "influenc": 34, "some": [34, 37, 38, 39], "util": [34, 37], "continu": 34, "zappi": 34, "what": 34, "interven": 34, "wasn": 34, "concern": 34, "less": 34, "daunt": 34, "And": 34, "better": [34, 39], "remot": 35, "below": [35, 37], "sometim": 35, "thread": [35, 39], "pythondagexecutor": 35, "intend": 35, "small": 35, "larger": [35, 39], "easiest": 35, "becaus": [35, 37], "handl": [35, 39], "automat": [35, 37, 39, 40], "sign": [35, 39], "free": 35, "account": 35, "300": 35, "slightli": 35, "variou": [35, 37], "far": [35, 39], "lambda": [35, 39], "docker": 35, "contain": [35, 37], "1000": 35, "style": 35, "rel": 35, "highest": [35, 37], "overhead": 35, "startup": [35, 39], "minut": 35, "compar": 35, "20": [35, 37], "therefor": 35, "much": [35, 37, 39], "modal_async": 35, "asyncmodaldagexecutor": 35, "s3": [35, 38, 40], "tomwhit": [35, 40], "temp": [35, 40], "2gb": [35, 37, 39], "altern": 35, "abov": [35, 39], "introduc": 36, "concept": 36, "help": [36, 37, 39], "out": [36, 39, 40], "delet": 36, "strong": [36, 39], "consist": 36, "retri": 36, "timeout": 36, "straggler": [36, 39], "prefac": 36, "theoret": 36, "v": 36, "practic": 36, "consider": 36, "diagnos": 36, "tip": 36, "ensur": [37, 40], "never": 37, "exce": 37, "illustr": 37, "diagaram": 37, "your": [37, 39], "machin": [37, 39], "precis": 37, "compress": 37, "conserv": 37, "upper": 37, "projected_mem": 37, "calcul": 37, "greater": 37, "except": [37, 38], "rais": 37, "phase": 37, "check": 37, "confid": 37, "within": 37, "budget": 37, "properli": 37, "measure_reserved_mem": [37, 39], "basi": 37, "baselin": 37, "accur": 37, "estim": 37, "abil": 37, "peak_measured_mem": 37, "actual": 37, "analys": 37, "ran": 37, "room": 37, "rule": [37, 40], "thumb": 37, "least": 37, "ten": 37, "decompress": 37, "basic": 37, "four": 37, "itself": 37, "complex": [37, 39], "particular": 37, "around": [37, 39], "good": [37, 39], "100mb": [37, 39], "factor": [37, 39], "smaller": 37, "plenti": 37, "fault": 38, "toler": 38, "section": 38, "cover": 38, "featur": 38, "fashion": 38, "reli": 38, "global": 38, "importantli": 38, "amazon": 38, "fail": 38, "again": 38, "whole": 38, "error": 38, "messag": 38, "take": [38, 39], "longer": [38, 39], "pre": 38, "determin": [38, 39], "consid": [38, 39], "paragraph": 38, "down": 38, "mitig": 38, "specul": 38, "duplic": 38, "launch": 38, "certain": [38, 40], "circumst": 38, "act": 38, "backup": [38, 39], "henc": [38, 39], "bring": 38, "overal": [38, 39], "taken": 38, "origin": 38, "cancel": 38, "expect": [38, 39], "ident": 38, "idempot": 38, "atom": 38, "updat": 38, "kei": 38, "experiment": 38, "disabl": 38, "maintain": 39, "terabyt": 39, "deeper": 39, "scenario": 39, "horizont": 39, "versu": 39, "vertic": 39, "throughput": 39, "upgrad": 39, "exist": 39, "speed": 39, "weak": 39, "solut": 39, "processor": 39, "fix": 39, "word": 39, "faster": 39, "done": 39, "big": 39, "dataset": 39, "affect": 39, "elementwis": 39, "ideal": 39, "infinit": 39, "concurr": 39, "linear": 39, "could": 39, "proportion": 39, "those": 39, "achiev": 39, "firstli": 39, "sure": [39, 40], "re": 39, "enough": 39, "might": 39, "necessari": 39, "adjust": 39, "restrict": 39, "max_work": 39, "With": 39, "fewer": 39, "wait": 39, "averag": 39, "who": 39, "hold": 39, "next": 39, "use_backup": 39, "failur": 39, "restart": 39, "becom": 39, "equal": 39, "minimum": 39, "signific": 39, "carri": 39, "iter": 39, "fuse": 39, "enhanc": 39, "cannot": 39, "potenti": 39, "violat": 39, "constraint": 39, "theori": 39, "cumul": 39, "independ": 39, "branch": 39, "simultan": 39, "suffici": 39, "logic": 39, "feed": 39, "vice": 39, "versa": 39, "necessarili": 39, "come": 39, "own": 39, "instanc": 39, "cluster": 39, "base": 39, "g": 39, "characterist": 39, "provis": 39, "offer": 39, "gcf": 39, "observ": 39, "view": 39, "point": 39, "want": 39, "line": 39, "plot": 39, "perfect": 39, "magic": 39, "long": 39, "suggest": 39, "stick": 39, "filesystem": 40, "By": 40, "temporari": 40, "appropri": 40, "region": 40, "bucket": 40, "doe": 40, "clear": 40, "space": 40, "incur": 40, "unnecessari": 40, "cost": 40, "typic": 40, "remov": 40, "old": 40, "job": 40, "short": 40, "period": 40, "manual": 40, "clean": 40, "tmpdir": 40, "regular": 40, "command": 40, "rm": 40, "On": 40, "conveni": 40, "dedic": 40, "lifecycl": 40, "consol": 40, "click": 40, "tab": 40, "ag": 40, "enter": 40, "dai": 40, "instruct": 40}, "objects": {"cubed": [[5, 0, 1, "", "Array"], [9, 0, 1, "", "Callback"], [10, 0, 1, "", "Spec"], [11, 0, 1, "", "TaskEndEvent"], [12, 2, 1, "", "apply_gufunc"], [13, 2, 1, "", "compute"], [14, 2, 1, "", "from_array"], [15, 2, 1, "", "from_zarr"], [16, 2, 1, "", "map_blocks"], [17, 2, 1, "", "measure_reserved_mem"], [18, 2, 1, "", "nanmean"], [19, 2, 1, "", "nansum"], [25, 2, 1, "", "store"], [26, 2, 1, "", "to_zarr"], [27, 2, 1, "", "visualize"]], "cubed.Array": [[5, 1, 1, "", "__init__"], [6, 1, 1, "", "compute"], [7, 1, 1, "", "rechunk"], [8, 1, 1, "", "visualize"]], "cubed.Callback": [[9, 1, 1, "", "__init__"]], "cubed.Spec": [[10, 1, 1, "", "__init__"]], "cubed.TaskEndEvent": [[11, 1, 1, "", "__init__"]], "cubed.array_api": [[1, 2, 1, "", "arange"], [1, 2, 1, "", "asarray"], [1, 2, 1, "", "broadcast_to"], [1, 2, 1, "", "empty"], [1, 2, 1, "", "empty_like"], [1, 2, 1, "", "eye"], [1, 2, 1, "", "full"], [1, 2, 1, "", "full_like"], [1, 2, 1, "", "linspace"], [1, 2, 1, "", "ones"], [1, 2, 1, "", "ones_like"], [1, 2, 1, "", "zeros"], [1, 2, 1, "", "zeros_like"]], "cubed.random": [[20, 2, 1, "", "random"]], "cubed.runtime.executors.beam": [[21, 0, 1, "", "BeamDagExecutor"]], "cubed.runtime.executors.beam.BeamDagExecutor": [[21, 1, 1, "", "__init__"]], "cubed.runtime.executors.lithops": [[22, 0, 1, "", "LithopsDagExecutor"]], "cubed.runtime.executors.lithops.LithopsDagExecutor": [[22, 1, 1, "", "__init__"]], "cubed.runtime.executors.modal_async": [[23, 0, 1, "", "AsyncModalDagExecutor"]], "cubed.runtime.executors.modal_async.AsyncModalDagExecutor": [[23, 1, 1, "", "__init__"]], "cubed.runtime.executors.python": [[24, 0, 1, "", "PythonDagExecutor"]], "cubed.runtime.executors.python.PythonDagExecutor": [[24, 1, 1, "", "__init__"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"]}, "titleterms": {"api": [0, 1, 4], "refer": 0, "arrai": [0, 1, 4, 5, 6, 7, 8, 32], "io": 0, "chunk": [0, 37], "specif": 0, "function": 0, "non": 0, "standardis": 0, "random": [0, 20], "number": 0, "gener": 0, "runtim": [0, 4, 21, 22, 23, 24], "executor": [0, 21, 22, 23, 24, 35, 39], "python": [1, 24, 35], "miss": 1, "from": 1, "cube": [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 39], "differ": [1, 39], "between": 1, "standard": 1, "comput": [2, 6, 13], "plan": [2, 39], "memori": [2, 32, 37], "execut": 2, "contribut": 3, "develop": [3, 32], "design": 4, "storag": [4, 40], "primit": 4, "oper": [4, 33], "core": 4, "rechunk": [7, 33], "visual": [8, 27, 39], "callback": [9, 39], "spec": 10, "taskendev": 11, "apply_gufunc": 12, "from_arrai": 14, "from_zarr": 15, "map_block": [16, 33], "measure_reserved_mem": 17, "nanmean": 18, "nansum": 19, "beam": 21, "beamdagexecutor": 21, "lithop": 22, "lithopsdagexecutor": 22, "modal_async": 23, "asyncmodaldagexecutor": 23, "pythondagexecutor": 24, "store": 25, "to_zarr": 26, "demo": 28, "get": 29, "start": 29, "instal": 30, "conda": 30, "pip": 30, "option": 30, "depend": [30, 33], "why": 31, "bound": 32, "serverless": 32, "distribut": 32, "n": 32, "dimension": 32, "process": 32, "document": 32, "For": 32, "user": [32, 36], "tree": 33, "elemwis": 33, "map_direct": 33, "blockwis": 33, "reduct": 33, "arg_reduct": 33, "relat": 34, "project": [34, 37], "dask": 34, "xarrai": 34, "previou": 34, "work": 34, "local": 35, "which": 35, "cloud": [35, 39, 40], "servic": 35, "should": 35, "i": 35, "us": 35, "specifi": 35, "an": 35, "guid": 36, "allow": 37, "reserv": 37, "size": 37, "reliabl": 38, "strong": 38, "consist": 38, "retri": 38, "timeout": 38, "straggler": 38, "scale": 39, "prefac": 39, "type": 39, "theoret": 39, "v": 39, "practic": 39, "singl": 39, "step": 39, "calcul": 39, "multi": 39, "pipelin": 39, "other": 39, "perform": 39, "consider": 39, "provid": 39, "diagnos": 39, "optim": 39, "histori": 39, "timelin": 39, "tip": 39, "delet": 40, "intermedi": 40, "data": 40}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 60}, "alltitles": {"API Reference": [[0, "api-reference"]], "Array": [[0, "array"]], "IO": [[0, "io"]], "Chunk-specific functions": [[0, "chunk-specific-functions"]], "Non-standardised functions": [[0, "non-standardised-functions"]], "Random number generation": [[0, "random-number-generation"]], "Runtime": [[0, "runtime"], [4, "runtime"]], "Executors": [[0, "executors"], [35, "executors"]], "Python Array API": [[1, "python-array-api"]], "Missing from Cubed": [[1, "missing-from-cubed"]], "Differences between Cubed and the standard": [[1, "differences-between-cubed-and-the-standard"]], "Computation": [[2, "computation"]], "Plan": [[2, "plan"]], "Memory": [[2, "memory"], [37, "memory"]], "Execution": [[2, "execution"]], "Contributing": [[3, "contributing"]], "Development": [[3, "development"]], "Design": [[4, "design"]], "Storage": [[4, "storage"], [40, "storage"]], "Primitive operations": [[4, "primitive-operations"]], "Core operations": [[4, "core-operations"]], "Array API": [[4, "array-api"]], "cubed.Array": [[5, "cubed-array"]], "cubed.Array.compute": [[6, "cubed-array-compute"]], "cubed.Array.rechunk": [[7, "cubed-array-rechunk"]], "cubed.Array.visualize": [[8, "cubed-array-visualize"]], "cubed.Callback": [[9, "cubed-callback"]], "cubed.Spec": [[10, "cubed-spec"]], "cubed.TaskEndEvent": [[11, "cubed-taskendevent"]], "cubed.apply_gufunc": [[12, "cubed-apply-gufunc"]], "cubed.compute": [[13, "cubed-compute"]], "cubed.from_array": [[14, "cubed-from-array"]], "cubed.from_zarr": [[15, "cubed-from-zarr"]], "cubed.map_blocks": [[16, "cubed-map-blocks"]], "cubed.measure_reserved_mem": [[17, "cubed-measure-reserved-mem"]], "cubed.nanmean": [[18, "cubed-nanmean"]], "cubed.nansum": [[19, "cubed-nansum"]], "cubed.random.random": [[20, "cubed-random-random"]], "cubed.runtime.executors.beam.BeamDagExecutor": [[21, "cubed-runtime-executors-beam-beamdagexecutor"]], "cubed.runtime.executors.lithops.LithopsDagExecutor": [[22, "cubed-runtime-executors-lithops-lithopsdagexecutor"]], "cubed.runtime.executors.modal_async.AsyncModalDagExecutor": [[23, "cubed-runtime-executors-modal-async-asyncmodaldagexecutor"]], "cubed.runtime.executors.python.PythonDagExecutor": [[24, "cubed-runtime-executors-python-pythondagexecutor"]], "cubed.store": [[25, "cubed-store"]], "cubed.to_zarr": [[26, "cubed-to-zarr"]], "cubed.visualize": [[27, "cubed-visualize"]], "Demo": [[28, "demo"]], "Getting Started": [[29, "getting-started"]], "Installation": [[30, "installation"]], "Conda": [[30, "conda"]], "Pip": [[30, "pip"]], "Optional dependencies": [[30, "optional-dependencies"]], "Why Cubed?": [[31, "why-cubed"]], "Cubed": [[32, "cubed"]], "Bounded-memory serverless distributed N-dimensional array processing": [[32, "bounded-memory-serverless-distributed-n-dimensional-array-processing"]], "Documentation": [[32, "documentation"]], "For users": [[32, null]], "For developers": [[32, null]], "Operations": [[33, "operations"]], "Dependency Tree": [[33, "dependency-tree"]], "elemwise": [[33, "elemwise"]], "map_blocks": [[33, "map-blocks"]], "map_direct": [[33, "map-direct"]], "blockwise": [[33, "blockwise"]], "rechunk": [[33, "rechunk"]], "reduction and arg_reduction": [[33, "reduction-and-arg-reduction"]], "Related Projects": [[34, "related-projects"]], "Dask": [[34, "dask"]], "Xarray": [[34, "xarray"]], "Previous work": [[34, "previous-work"]], "Local Python executor": [[35, "local-python-executor"]], "Which cloud service should I use?": [[35, "which-cloud-service-should-i-use"]], "Specifying an executor": [[35, "specifying-an-executor"]], "User Guide": [[36, "user-guide"]], "Allowed memory": [[37, "allowed-memory"]], "Projected memory": [[37, "projected-memory"]], "Reserved memory": [[37, "reserved-memory"]], "Chunk sizes": [[37, "chunk-sizes"]], "Reliability": [[38, "reliability"]], "Strong consistency": [[38, "strong-consistency"]], "Retries": [[38, "retries"]], "Timeouts": [[38, "timeouts"]], "Stragglers": [[38, "stragglers"]], "Scaling": [[39, "scaling"]], "Preface: Types of Scaling": [[39, "preface-types-of-scaling"]], "Theoretical vs Practical Scaling of Cubed": [[39, "theoretical-vs-practical-scaling-of-cubed"]], "Single-step Calculation": [[39, "single-step-calculation"]], "Multi-step Calculation": [[39, "multi-step-calculation"]], "Multi-pipeline Calculation": [[39, "multi-pipeline-calculation"]], "Other Performance Considerations": [[39, "other-performance-considerations"]], "Different Executors": [[39, "different-executors"]], "Different Cloud Providers": [[39, "different-cloud-providers"]], "Diagnosing Performance": [[39, "diagnosing-performance"]], "Optimized Plan": [[39, "optimized-plan"]], "History Callback": [[39, "history-callback"]], "Timeline Visualization Callback": [[39, "timeline-visualization-callback"]], "Tips": [[39, "tips"]], "Cloud storage": [[40, "cloud-storage"]], "Deleting intermediate data": [[40, "deleting-intermediate-data"]]}, "indexentries": {"arange() (in module cubed.array_api)": [[1, "cubed.array_api.arange"]], "asarray() (in module cubed.array_api)": [[1, "cubed.array_api.asarray"]], "broadcast_to() (in module cubed.array_api)": [[1, "cubed.array_api.broadcast_to"]], "empty() (in module cubed.array_api)": [[1, "cubed.array_api.empty"]], "empty_like() (in module cubed.array_api)": [[1, "cubed.array_api.empty_like"]], "eye() (in module cubed.array_api)": [[1, "cubed.array_api.eye"]], "full() (in module cubed.array_api)": [[1, "cubed.array_api.full"]], "full_like() (in module cubed.array_api)": [[1, "cubed.array_api.full_like"]], "linspace() (in module cubed.array_api)": [[1, "cubed.array_api.linspace"]], "ones() (in module cubed.array_api)": [[1, "cubed.array_api.ones"]], "ones_like() (in module cubed.array_api)": [[1, "cubed.array_api.ones_like"]], "zeros() (in module cubed.array_api)": [[1, "cubed.array_api.zeros"]], "zeros_like() (in module cubed.array_api)": [[1, "cubed.array_api.zeros_like"]], "array (class in cubed)": [[5, "cubed.Array"]], "__init__() (cubed.array method)": [[5, "cubed.Array.__init__"]], "compute() (cubed.array method)": [[6, "cubed.Array.compute"]], "rechunk() (cubed.array method)": [[7, "cubed.Array.rechunk"]], "visualize() (cubed.array method)": [[8, "cubed.Array.visualize"]], "callback (class in cubed)": [[9, "cubed.Callback"]], "__init__() (cubed.callback method)": [[9, "cubed.Callback.__init__"]], "spec (class in cubed)": [[10, "cubed.Spec"]], "__init__() (cubed.spec method)": [[10, "cubed.Spec.__init__"]], "taskendevent (class in cubed)": [[11, "cubed.TaskEndEvent"]], "__init__() (cubed.taskendevent method)": [[11, "cubed.TaskEndEvent.__init__"]], "apply_gufunc() (in module cubed)": [[12, "cubed.apply_gufunc"]], "compute() (in module cubed)": [[13, "cubed.compute"]], "from_array() (in module cubed)": [[14, "cubed.from_array"]], "from_zarr() (in module cubed)": [[15, "cubed.from_zarr"]], "map_blocks() (in module cubed)": [[16, "cubed.map_blocks"]], "measure_reserved_mem() (in module cubed)": [[17, "cubed.measure_reserved_mem"]], "nanmean() (in module cubed)": [[18, "cubed.nanmean"]], "nansum() (in module cubed)": [[19, "cubed.nansum"]], "random() (in module cubed.random)": [[20, "cubed.random.random"]], "beamdagexecutor (class in cubed.runtime.executors.beam)": [[21, "cubed.runtime.executors.beam.BeamDagExecutor"]], "__init__() (cubed.runtime.executors.beam.beamdagexecutor method)": [[21, "cubed.runtime.executors.beam.BeamDagExecutor.__init__"]], "lithopsdagexecutor (class in cubed.runtime.executors.lithops)": [[22, "cubed.runtime.executors.lithops.LithopsDagExecutor"]], "__init__() (cubed.runtime.executors.lithops.lithopsdagexecutor method)": [[22, "cubed.runtime.executors.lithops.LithopsDagExecutor.__init__"]], "asyncmodaldagexecutor (class in cubed.runtime.executors.modal_async)": [[23, "cubed.runtime.executors.modal_async.AsyncModalDagExecutor"]], "__init__() (cubed.runtime.executors.modal_async.asyncmodaldagexecutor method)": [[23, "cubed.runtime.executors.modal_async.AsyncModalDagExecutor.__init__"]], "pythondagexecutor (class in cubed.runtime.executors.python)": [[24, "cubed.runtime.executors.python.PythonDagExecutor"]], "__init__() (cubed.runtime.executors.python.pythondagexecutor method)": [[24, "cubed.runtime.executors.python.PythonDagExecutor.__init__"]], "store() (in module cubed)": [[25, "cubed.store"]], "to_zarr() (in module cubed)": [[26, "cubed.to_zarr"]], "visualize() (in module cubed)": [[27, "cubed.visualize"]]}})