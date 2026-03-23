import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier


#inicializa o mlflow
mlflow.start_run()

#Define o modelo
clf = RandomForestClassifier(n_estimators=100)

#Treina o modelo
clf.fit(X_train, y_train)

#Loga as métricas do modelo
mlflow.log_metric("accuracy", clf.score(X_test, y_test))

#Salva o modelo com o mlflow
mlflow.sklearn.log_model(clf, "random-forest-model")

#finaliza o run
mlflow.end_run()
