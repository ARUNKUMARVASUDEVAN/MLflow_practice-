import mlflow

def calculate(x,y):
    return x+y




if __name__ =='__main__':
    with mlflow.start_run():
        x,y=10,20
        z=calculate(x,y)
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)