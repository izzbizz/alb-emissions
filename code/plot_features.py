import shap
import matplotlib.pyplot as plt

def plot_features(clf, df, outdir):
    '''plot the top 20 predictors for the model
       input: trained model, complete pandas dataframe'''
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df.drop('TARGET', axis=1))
    shap.summary_plot(shap_values, df.drop('TARGET', axis=1), plot_type="bar", show=False)
    plt.savefig(outdir + 'top_features_aggregated.png', bbox_inches='tight')
    shap.summary_plot(shap_values, df.drop('TARGET', axis=1), show=False)
    plt.savefig(outdir + 'top_features.png', bbox_inches='tight')    
