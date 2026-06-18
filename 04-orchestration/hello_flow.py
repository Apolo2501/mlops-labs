from prefect import flow, task

@task
def say_hello():
    print("Hola Antonio, Prefect está funcionando!")

@flow
def hello_flow():
    say_hello()

if __name__ == "__main__":
    hello_flow()