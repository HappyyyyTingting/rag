#选择基础镜像
FROM python:3.10-slim

#设置工作目录
WORKDIR /app

#复制依赖文件
COPY requirements.txt /app/

#安装依赖
RUN pip install --no-cached-dir -r requirements.txt

#复制项目文件
COPY  . /app

#定义容器启动时运行的命令
CMD ["python", "chat.py"]