import joblib
import numpy as np
import os

linear_model = joblib.load('linear.joblib')
logistic_model = joblib.load('logistic.joblib')

linear_code = """
float linear_regression_prediction(float* features, float* thetas, int n_thetas ) 
{
        float r = thetas[0];
        for (int i = 1; i < n_thetas-1; i++)
        {
            r += features[i-1] * thetas[i];
        }
        return r;
}
"""

logistic_code = """
float logistic_regression(float* features, float* thetas, int n_parameter) {
    
    float linear_regression_prediction(float* features, float* thetas, int n_thetas ) 
    {
        float r = thetas[0];
        for (int i = 1; i < n_thetas-1; i++)
        {
            r += features[i-1] * thetas[i];
        }
        return r;
    }

    float sigmoid(float x) {
        return 1.f / (1.f + exp_approx(-x, 10));
    }
    
    float res = sigmoid(linear_regression_prediction(features, thetas, n_parameter));
    
    return res > 0.5f ? 1.f : 0.f;
}
"""

lin_thetas = [linear_model.intercept_]
lin_thetas += linear_model.coef_.tolist()

log_thetas = [logistic_model.intercept_[0]]
log_thetas += logistic_model.coef_[0].tolist()

n_thetas_lin = len(lin_thetas)
n_thetas_log = len(log_thetas)

lin_thetas = ('f,'.join(np.array(lin_thetas).astype(str))).strip(',')
log_thetas = ('f,'.join(np.array(log_thetas).astype(str))).strip(',')

lin_features = (np.linspace(1,10,n_thetas_lin-1)).tolist()
log_features = (np.linspace(1,10,n_thetas_log-1)).tolist()

lin_features = ('f,'.join(np.array(lin_features).astype(str))).strip(',')
log_features = ('f,'.join(np.array(log_features).astype(str))).strip(',')

lin_main = f"""
        { linear_code }

        int main(int argc, char* argv[])
        {{
            float features[{n_thetas_lin}-1] = {{ {lin_features} }};
            float thetas[{n_thetas_lin}] = {{ {lin_thetas} }};
            return linear_regression_prediction(features, thetas, {n_thetas_lin});
        }}
"""

log_main = f"""
        { logistic_code }

        int main(int argc, char* argv[])
        {{
            float features[{n_thetas_log}-1] = {{ {log_features} }};
            float thetas[{n_thetas_log}] = {{ {log_thetas} }};
            return logistic_regression(features, thetas, {n_thetas_log});
        }}
"""

linf = open("linear.c", "w")
linf.write(lin_main)

logf = open("logistic.c", "w")
logf.write(log_main)

os.system("gcc linear.c")
os.system("gcc logistic.c")