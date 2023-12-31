# Bazel


> [Bazel](https://docs.bazel.build/versions/master/build-ref.html) builds software from source code organized in a directory called a workspace. Source files in the workspace are organized in a nested hierarchy of packages, where each package is a directory that contains a set of related source files and one BUILD file. The BUILD file specifies what software outputs can be built from the source.

## 1.Introduce

### 1.1. Features

- High-level build language
- fast and reliable
- multi-platform
- large scales and extensible

### 1.2. workflow

1. **Set up Bazel.** Download and [install Bazel](https://docs.bazel.build/versions/master/install.html).

2. **Set up a project [workspace](https://docs.bazel.build/versions/master/build-ref.html#workspaces)**, which is a directory where Bazel looks for build inputs and `BUILD` files, and where it stores build outputs.

3. **Write a `BUILD` file**, which tells Bazel what to build and how to build it.

   You write your `BUILD` file by declaring build targets using [Starlark](https://docs.bazel.build/versions/master/skylark/language.html), a domain-specific language. (See example [here](https://github.com/bazelbuild/bazel/blob/master/examples/cpp/BUILD).)

   A build target specifies a set of input artifacts that Bazel will build plus their dependencies, the build rule Bazel will use to build it, and options that configure the build rule.

   A build rule specifies the build tools Bazel will use, such as compilers and linkers, and their configurations. Bazel ships with a number of build rules covering the most common artifact types in the supported languages on supported platforms.

4. **Run Bazel** from the [command line](https://docs.bazel.build/versions/master/command-line-reference.html). Bazel places your outputs within the workspace.

### 1.3. backend workflow

1. **Loads** the `BUILD` files relevant to the target.
2. **Analyzes** the inputs and their [dependencies](https://docs.bazel.build/versions/master/build-ref.html#dependencies), applies the specified build rules, and produces an [action](https://docs.bazel.build/versions/master/skylark/concepts.html#evaluation-model) graph.
3. **Executes** the build actions on the inputs until the final build outputs are produced.

Since all previous build work is cached, Bazel can identify and reuse cached artifacts and only rebuild or retest what’s changed. To further enforce correctness, you can set up Bazel to run builds and tests [hermetically](https://docs.bazel.build/versions/master/guide.html#sandboxing) through sandboxing, minimizing skew and maximizing [reproducibility](https://docs.bazel.build/versions/master/guide.html#correctness).

### 1.4. Download

```bash
sudo apt install g++ unzip zip
# Ubuntu 16.04 (LTS) uses OpenJDK 8 by default:
sudo apt-get install openjdk-8-jdk

# Ubuntu 18.04 (LTS) uses OpenJDK 11 by default:
sudo apt-get install openjdk-11-jdk
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-<version>-installer-linux-x86_64.sh --user
```

> If you ran the Bazel installer with the `--user` flag as above, the Bazel executable is installed in your `$HOME/bin` directory. It’s a good idea to add this directory to your default paths, as follows:

```bash
export PATH="$PATH:$HOME/bin"
```

## 2. Example

### 2.1. Structure&SingleBuild

the CPP examples structure

```
examples
└── cpp-tutorial
    ├──stage1
    │  ├── main
    │  │   ├── BUILD
    │  │   └── hello-world.cc
    │  └── WORKSPACE
    ├──stage2
    │  ├── main
    │  │   ├── BUILD
    │  │   ├── hello-world.cc
    │  │   ├── hello-greet.cc
    │  │   └── hello-greet.h
    │  └── WORKSPACE
    └──stage3
       ├── main
       │   ├── BUILD
       │   ├── hello-world.cc
       │   ├── hello-greet.cc
       │   └── hello-greet.h
       ├── lib
       │   ├── BUILD
       │   ├── hello-time.cc
       │   └── hello-time.h
       └── WORKSPACE
```

- **workspace:** a directory that holds your project’s source files and Bazel’s build outputs. It also contains files that Bazel recognizes as special:
  - The `WORKSPACE` file, which identifies the directory and its contents as a Bazel workspace and lives at the root of the project’s directory structure,
  - One or more `BUILD` files, which tell Bazel how to build different parts of the project. (A directory within the workspace that contains a `BUILD` file is a *package*. You will learn about packages later in this tutorial.)

- **Build File:** 

```bash
cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)
```

> In our example, the `hello-world` target instantiates Bazel’s built-in [`cc_binary` rule](https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary). The rule tells Bazel to build a self-contained executable binary from the `hello-world.cc` source file with no dependencies.

- **Build Project: **

```bash
bazel build //main:hello-world # in the stage1//
```

> `//main:` part is the location of our `BUILD` file relative to the root of the workspace, and `hello-world` is what we named that target in the `BUILD` file

- **Review dependency graph:** 

```bash
bazel query --notool_deps --noimplicit_deps "deps(//main:hello-world)" \
  --output graph
```

> to look for all dependencies for the target `//main:hello-world` (excluding host and implicit dependencies) and format the output as a graph.

```bash
# view the graph locally by installing GraphViz and the xdot Dot Viewer:
sudo apt update && sudo apt install graphviz xdot
# generate and view the graph by piping the text output above straight to xdot:
xdot <(bazel query --notool_deps --noimplicit_deps "deps(//main:hello-world)" \
  --output graph)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200828200445.png)

### 2.2. Multi-build

```
cc_library(
    name = "hello-greet",
    srcs = ["hello-greet.cc"],
    hdrs = ["hello-greet.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-greet",
    ],
)
```

> he `deps` attribute in the `hello-world` target tells Bazel that the `hello-greet` library is required to build the `hello-world` binary.

### 2.3. Multi-Package

Take a look at the `lib/BUILD` file:

```
cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
    visibility = ["//main:__pkg__"],
)
```

And at the `main/BUILD` file:

```
cc_library(
    name = "hello-greet",
    srcs = ["hello-greet.cc"],
    hdrs = ["hello-greet.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-greet",
        "//lib:hello-time",
    ],
)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200828201159.png)

> //path//to/package: target-name
>
> If the target is a rule target, then `path/to/package` is the path to the directory containing the `BUILD` file, and `target-name` is what you named the target in the `BUILD` file (the `name` attribute). If the target is a file target, then `path/to/package` is the path to the root of the package, and `target-name` is the name of the target file, including its full path.

## 3. Common rules

### 3.1. including multiple files in a target

```bash
cc_library(
    name = "build-all-the-files",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
)
```

### 3.2. Using transitive includes

> If a file includes a header, then the file’s rule should depend on that header’s library. Conversely, only direct dependencies need to be specified as dependencies. For example, suppose `sandwich.h` includes `bread.h` and `bread.h` includes `flour.h`. `sandwich.h` doesn’t include `flour.h` (who wants flour in their sandwich?), so the `BUILD` file would look like this:

```bash
cc_library(
    name = "sandwich",
    srcs = ["sandwich.cc"],
    hdrs = ["sandwich.h"],
    deps = [":bread"],
)
cc_library(
    name = "bread",
    srcs = ["bread.cc"],
    hdrs = ["bread.h"],
    deps = [":flour"],
)
cc_library(
    name = "flour",
    srcs = ["flour.cc"],
    hdrs = ["flour.h"],
)
```

### 3.3. Adding including paths

```bash
└── my-project
    ├── legacy
    │   └── some_lib
    │       ├── BUILD
    │       ├── include
    │       │   └── some_lib.h
    │       └── some_lib.cc
    └── WORKSPACE
```

```bash
cc_library(
    name = "some_lib",
    srcs = ["some_lib.cc"],
    hdrs = ["include/some_lib.h"],
    copts = ["-Ilegacy/some_lib/include"],
)
```

> copts 这个参数含义不太明白

### 3.4. Adding dependencies on precompiled libraries

```bash
cc_library(
    name = "mylib",
    srcs = ["mylib.so"],
    hdrs = ["mylib.h"],
)
```

### 3.5. including external libraries

```bash
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "@//:gtest.BUILD",
)
```

```bash
cc_library(
    name = "main",
    srcs = glob(
        ["googletest-release-1.7.0/src/*.cc"],
        exclude = ["googletest-release-1.7.0/src/gtest-all.cc"]
    ),
    hdrs = glob([
        "googletest-release-1.7.0/include/**/*.h",
        "googletest-release-1.7.0/src/*.h"
    ]),
    copts = [
        "-Iexternal/gtest/googletest-release-1.7.0/include",
        "-Iexternal/gtest/googletest-release-1.7.0"
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
```

```
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "@//:gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)
```

```bash
cc_library(
    name = "main",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/gtest-all.cc"]
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h"
    ]),
    copts = ["-Iexternal/gtest/include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
```

#### 3.5.1 depending on other bazel projects

> If you want to use targets from a second Bazel project, you can use [`local_repository`](http://docs.bazel.build/be/workspace.html#local_repository), [`git_repository`](https://docs.bazel.build/versions/master/repo/git.html#git_repository) or [`http_archive`](https://docs.bazel.build/versions/master/repo/http.html#http_archive) to symlink it from the local filesystem, reference a git repository or download it (respectively).

```bash
local_repository(
    name = "coworkers_project",
    path = "/path/to/coworkers-project",
)
```

#### 3.5.2.depending on non-bazel projects

> Rules prefixed with `new_`, e.g., [`new_local_repository`](http://docs.bazel.build/be/workspace.html#new_local_repository), allow you to create targets from projects that do not use Bazel.

```bash
new_local_repository(
    name = "coworkers_project",
    path = "/path/to/coworkers-project",
    build_file = "coworker.BUILD",
)
```

### 3.6. Shadowing dependencies

```bash
workspace(name = "myproject")
local_repository(
    name = "A",
    path = "../A",
)
local_repository(
    name = "B",
    path = "../B",
)
```

A/WORKSPACE

```bash
workspace(name = "A")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner",
    urls = ["https://github.com/testrunner/v1.zip"],
    sha256 = "...",
)
```

B/WORKSPACE

```bash
workspace(name = "B")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner",
    urls = ["https://github.com/testrunner/v2.zip"],
    sha256 = "..."
)
```

the above confront verson differ problem which can be solved by follows:

```bash
workspace(name = "myproject")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner-v1",
    urls = ["https://github.com/testrunner/v1.zip"],
    sha256 = "..."
)
http_archive(
    name = "testrunner-v2",
    urls = ["https://github.com/testrunner/v2.zip"],
    sha256 = "..."
)
local_repository(
    name = "A",
    path = "../A",
    repo_mapping = {"@testrunner" : "@testrunner-v1"}
)
local_repository(
    name = "B",
    path = "../B",
    repo_mapping = {"@testrunner" : "@testrunner-v2"}
)
```

### 3.7. Visibility

- `"//visibility:public"`: Anyone can use this target. (May not be combined with any other specification.)
- `"//visibility:private"`: Only targets in this package can use this target. (May not be combined with any other specification.)
- `"//foo/bar:__pkg__"`: Grants access to targets defined in `//foo/bar` (but not its subpackages). Here, `__pkg__` is a special piece of syntax representing all of the targets in a package.
- `"//foo/bar:__subpackages__"`: Grants access to targets defined in `//foo/bar`, or any of its direct or indirect subpackages. Again, `__subpackages__` is special syntax.
- `"//foo/bar:my_package_group"`: Grants access to all of the packages named by the given [package group](https://docs.bazel.build/versions/master/be/functions.html#package_group).
  - Package groups do not support the special `__pkg__` and `__subpackages__` syntax. Within a package group, `"//foo/bar"` is equivalent to `"//foo/bar:__pkg__"` and `"//foo/bar/..."` is equivalent to `"//foo/bar:__subpackages__"`.

> `load` statements are currently not subject to visibility. It is possible to load a `bzl` file anywhere in the workspace.



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/bazel/  

