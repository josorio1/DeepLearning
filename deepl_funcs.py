from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt




def confusion_matrix_plot_nonbin(y_true,y_pred):
    
    """
    Provided with true and predicted labels, it returns its confusion matrix.
    """
    cm = confusion_matrix(y_true,y_pred)

    df_cm = pd.DataFrame(cm) #index and column are given as 0-9 by default in DataFrame, so we don't need to add it 

    plt.figure(figsize=(12,6))
    sn.heatmap(df_cm, annot=True,fmt="d",  cmap=plt.cm.Blues, annot_kws={"size": 8}) # font size
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    plt.show()