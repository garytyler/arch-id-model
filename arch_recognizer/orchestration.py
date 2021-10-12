import asyncio
import os
import sys
import uuid

import docker
from docker import DockerClient
from docker.models.containers import Container


def docker_client() -> DockerClient:
    return DockerClient(base_url="unix:///var/run/docker.sock")


async def tf_container(
    # monkeypatch,
    docker_client,
    # settings,
    # pg_test_db_url,
    # pg_test_container_name,
    run_name,
    # params,
) -> Container:

    # _tortoise_config = db.get_tortoise_config()

    # def get_test_tortoise_config():
    #     nonlocal _tortoise_config
    #     _tortoise_config["connections"]["default"] = pg_test_db_url
    #     return _tortoise_config
    ar_container = docker_client.containers.get("devcontainer")
    print(ar_container)
    # monkeypatch.setattr(db, "get_tortoise_config", get_test_tortoise_config)

    orch_container=docker_client.containers.get("devcontainer"),
    orch_container_uid=ar_container.exec_run(["id", "-u"]).output.decode().strip(),

    output = ""
    container = docker_client.containers.run(
        image="tensorflow/tensorflow:latest-gpu-jupyter",

        # image="tensorflow/tensorflow:latest",
        # image=ar_container,
        name=f"training-{run_name}",
        # command=["python", "train/__main__.py"],
        command=["python", "-c", "print('Hello world')"],
        user=os.geteuid() if os.geteuid() == orch_container_uid else 0,
        working_dir=ar_container.attrs["Config"]["WorkingDir"],
        network_mode=f"container:{ar_container.id}",
        volumes_from=ar_container.id,
        ipc_mode="host",
        auto_remove=True,
        detach=True,
        stream=True,
        tty=False,
    )
    for line in container.logs(stream=True, stdout=True, stderr=True):
        output += line.decode()
        print(output)


async def main():
    docker_client = docker.from_env()
    # ar_container = docker_client.containers.get("devcontainer")
    # print(ar_container)
    await tf_container(docker_client, run_name="asdf")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


# pg_prod_container = docker_client.containers.get("postgres")
# pg_test_container = docker_client.containers.run(
#     name=pg_test_container_name,
#     image=pg_prod_container.image,
#     command=["postgres", "-c", "log_statement=all"],
#     environment=dict(
#         POSTGRES_HOST=pg_test_container_name,
#         POSTGRES_USER=settings.POSTGRES_USER,
#         POSTGRES_DB=settings.POSTGRES_DB,
#         POSTGRES_HOST_AUTH_METHOD="trust",
#     ),
#     network=pg_prod_container.attrs["HostConfig"]["NetworkMode"],
#     publish_all_ports=True,
#     detach=True,
#     remove=True,
#     auto_remove=True,
# )
# try:
#     while True:
#         response = pg_test_container.exec_run(
#             [
#                 "pg_isready",
#                 f"--host={pg_test_container_name}",
#                 f"--dbname={settings.POSTGRES_DB}",
#                 f"--username={settings.POSTGRES_USER}",
#             ]
#         )
#         if response.exit_code == 0:
#             break
#     yield pg_test_container
# finally:
#     pg_test_container.remove(v=True, force=True)


# def test_e2e_from_frontend(docker_client, pytestconfig, capsys):
#     frontend_container = docker_client.containers.get("frontend")
#     frontend_uid = frontend_container.exec_run(["id", "-u"]).output.decode().strip()
#     playwright_lib_version = (
#         frontend_container.exec_run(["npm", "view", "playwright", "version"])
#         .output.decode()
#         .strip()
#     )
#     output = ""
#     try:
#         pw_container = docker_client.containers.run(
#             image=f"mcr.microsoft.com/playwright:v{playwright_lib_version}-focal",
#             name=f"test-playwright-{uuid.uuid4()}",
#             command=["npm", "run", "coverage:e2e"],
#             user=os.geteuid() if os.geteuid() == frontend_uid else 0,
#             working_dir=frontend_container.attrs["Config"]["WorkingDir"],
#             network_mode=f"container:{frontend_container.id}",
#             volumes_from=frontend_container.id,
#             ipc_mode="host",
#             auto_remove=True,
#             detach=True,
#             stream=True,
#             tty=False,
#         )
#         for line in pw_container.logs(stream=True, stdout=True, stderr=True):
#             output += line.decode()
#         if 0 != pw_container.wait()["StatusCode"]:
#             raise EndToEndTestError(output)
#     finally:
#         try:
#             pw_container.remove(force=True)
#         except Exception:
#             pass
#     if output and pytestconfig.getoption("verbose") > 0:
#         with capsys.disabled():
#             sys.stdout.write(output)
