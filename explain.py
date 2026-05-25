import shap

def explain(model, X):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)