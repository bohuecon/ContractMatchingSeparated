
## log in

```
ssh ubuntu@117.50.187.184
```

## install julia

```
mkdir julia
wget https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz
tar zxvf julia-1.7.3-linux-x86_64.tar.gz 
```

Add julia to the path. To add Julia's bin folder (with full path) to PATH environment variable, you can edit the ~/.bashrc (or ~/.bash_profile) file. Open the file in your favourite editor and add a new line as follows:

```
export PATH="$PATH:~/julia/julia-1.7.3/bin"
```

Then run 
```
source ~/.bashrc
```

Now you should be able to run julia by typing directly. 

## change julia update source

这里提供一种针对 Julia 的全平台通用的方式： 
```
$JULIA_DEPOT_PATH/config/startup.jl ( 默认为 ~/.julia/config/startup.jl ) 
```
文件定义了每次启动 Julia 时都会执行的命令 as follows. If you do not have the directory and file, use mkdir and touch to create them.

```
# 每次打开 Julia 都自动切换到最近的 pkg server
if VERSION >= v"1.4"
    try
        using PkgServerClient
    catch e
        @warn "error while importing PkgServerClient" e
    end
end
```

Then you can start Julia and type versioninfo() to check. 

## add packages

```
add DataFrames, CSV, Plots, LaTeXStrings, BlackBoxOptim, Parameters, Distributions, Optim, JLD, HDF5, LinearAlgebra, Interpolations, Roots, Statistics
```

## screen

If you run a program in SSH, and then close out ssh you can not get back into the console.

However you can use screen to attach + detach a console.

If you have centos, run
yum -y install screen

If you have debian/ubuntu run
apt-get install screen

Once installed, screen is simple to use. Type

screen

run the command you want to run, for example

./run_server.sh

to detach run: ctrl + a + d

Once detached you can check current screens with

screen -ls

Use screen -r to attach a single screen.

On multiple screens you may see something like:

screen -ls
There are screens on:
119217.pts-3.webhosting1200 (Detached)
344074.pts-13.webhosting1200 (Detached)
825035.pts-1.webhosting1200 (Detached)
650824.downlbk (Detached)
4 Sockets in /var/run/screen/S-root.

In this case attach with the value before pts, for example

screen -r 344074

You can shorthand with screen -r 3 (assuming only one begins with the number 3).




