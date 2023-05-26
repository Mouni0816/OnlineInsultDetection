# OnlineInsultDetection


Data from Kaggle Competition :
https://www.kaggle.com/competitions/detecting-insults-in-social-commentary/overview 

Ce qui a été fait :
* Préparation : 
    * Nettoyage des commentaires
    * Analyse du nombre de mots 
    * Analyse dataset imbalanced + proposition solution
* NN :
    * Modèle baseline avec dataset train complet : Validation ACC (0.8017), Test ACC (0.7875), Insultant Précision (0.63), Insultant Recall (0.56)
    * Modèle baseline avec dataset undersampling : Validation ACC (0.7873), Test ACC (0.7926), Insultant Précision (0.61), Insultant Recall (0.70)
    * Modèle baseline avec un layer flatten undersampling
* CNN :
    * Modèle baseline avec dataset train complet : Validation ACC (0.8000), Test ACC (0.7909), Insultant Précision (0.63), Insultant Recall (0.59)
    * Modèle baseline avec dataset undersampling : Validation ACC (0.7899), Test ACC (0.7892), Insultant Précision (0.60), Insultant Recall (0.71)
    * Fine tuning CNN avec dataset train complet (4H, RandomSearchCV) : 'vocab_size': 9801, 'num_filters': 64, 'maxlen': 200, 'kernel_size': 7, 'embedding_dim': 50
        * Best modèle : Validation ACC (0.8093), Test ACC (0.8061), Insultant Précision (0.69), Insultant Recall (0.55)
    * Fine tuning CNN avec dataset undersampling (??H, RandomSearchCV) : 
        * Best modèle : Validation ACC (xx), Test ACC (xx), Insultant Précision (xx), Insultant Recall (xx)