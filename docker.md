# run container without volume
docker run -p 8888:8888 jupyter/scipy-notebook:33add21fab64

# run container with volume
docker run -v /home/rustem/trainee:/home/jovyan/ -p 8888:8888 jupyter/scipy-notebook:33add21fab64

# run container with volume and dependencies from Dockerfile
docker build -t my_notebook .
docker run -v /home/rustem/trainee:/home/jovyan/ -p 8888:8888 my_notebook

# into container
docker exec -it f3a84af9a8a8 bash

# copy into running container
docker cp /home/rustem/trainee/reports/4/gnn.ipynb f3a84af9a8a8:/home/jovyan/gnn.ipynb
