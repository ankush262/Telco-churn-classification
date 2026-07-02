from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
def evalute_model(model,X_test,y_test):
    """
    evalates an xgboost model on test data
    arguments:
              model : trained xgboost model
              X_test : test features
                y_test : test target
                """
    preds = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))