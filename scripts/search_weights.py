import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, accuracy_score, log_loss
import numpy as np
from scipy.special import softmax
def sigmoid(x):
    return 1/(1 + np.exp(-x))
pd.set_option('display.max_columns', 300)
import warnings
warnings.simplefilter('ignore')
from pdb import set_trace as st
from scipy.optimize import minimize

def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df


def score_(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pd.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
#     if submission.min().min() < 0:
#         raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )

    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    any_injury_loss = log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )

    label_group_losses.append(any_injury_loss)
    return label_group_losses

# Assign the appropriate weights to each category
def create_training_solution(y_train):
    sol_train = y_train.copy()
    # bowel healthy|injury sample weight = 1|2
    sol_train['bowel_weight'] = np.where(sol_train['bowel_injury'] == 1, 2, 1)
    # extravasation healthy/injury sample weight = 1|6
    sol_train['extravasation_weight'] = np.where(sol_train['extravasation_injury'] == 1, 6, 1)
    # kidney healthy|low|high sample weight = 1|2|4
    sol_train['kidney_weight'] = np.where(sol_train['kidney_low'] == 1, 2, np.where(sol_train['kidney_high'] == 1, 4, 1))
    # liver healthy|low|high sample weight = 1|2|4
    sol_train['liver_weight'] = np.where(sol_train['liver_low'] == 1, 2, np.where(sol_train['liver_high'] == 1, 4, 1))
    # spleen healthy|low|high sample weight = 1|2|4
    sol_train['spleen_weight'] = np.where(sol_train['spleen_low'] == 1, 2, np.where(sol_train['spleen_high'] == 1, 4, 1))
    # any healthy|injury sample weight = 1|6
    sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 1)
    return sol_train



if __name__ == "__main__":
    meta = pd.read_csv('../input/train_series_meta.csv')
    meta['image_id'] = meta['patient_id'].astype(str) + '_' + meta['series_id'].astype(str)
    good_image_ids = []
    for i, idf in meta.groupby('patient_id'):
        if len(idf)==1:
            good_image_ids.append(idf['image_id'].values[0])
            continue
        good_image_ids.append(idf[idf.aortic_hu==idf.aortic_hu.min()]['image_id'].values[0])

    targets = ['bowel_injury','extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high']
    configs = [
        'class0_masked_192_2segmodels_pad0',
        'class0_masked_192_2segmodels_pad0_caformer',
        'class23_2segmodels_v3_xcit_small',
        'class0_gru_chaug_256_2segmodels_v3',
        'class1_lstm_112_2segmodels_20epochs_auc_v3_pretrained2',
        'class4_pretrain_288_n25_2segmodels_v3',
        'class4_pretrain_288_n25_2segmodels_25epochs_v3',
        'class23_maxvit_224_2segmodels_v4_pretrained',
        'class4_pretrain_288_n25_2segmodels_v4',
        'class4_pretrain_288_n25_2segmodels_25epochs_v4',
        'class_all_pad30_maxvit_pretrained_class1',
        'class_all_pad30_maxvit_crop2_pretrained_class0',
        'class_all_pad30_convnext_pretrained_class1',
        'class_all_pad30_maxvit_pretrained_class23',
        'class_all_pad30_caformer_class23',
        'class_all_pad30_seresnext_crop2_pretrained_class4',
    ]

    train = pd.read_csv('../input/train2.5d_2models_th0.2_pad_30_25.csv')
    fold = pd.read_csv('../input/multi_label_stratified_folds.csv')
    del train['fold']
    train = train.merge(fold, on='patient_id')

    train = train.drop_duplicates('image_id')

    features = []
    count = 0
    for config in configs:
        if config == 'tmp':
            oof =pd.read_csv('tmp.csv')
            pred_cols = [f'pred_{col}' for col in targets if f'pred_{col}' in list(oof)]
        else:
            oof = pd.concat([pd.read_csv(f'../results/{config}/oof_fold{fold}.csv') for fold in range(4)])
            pred_cols = [f'pred_{col}' for col in targets if f'pred_{col}' in list(oof)]
            oof[pred_cols] = sigmoid(oof[pred_cols].values)
        for col in pred_cols:
            oof[col] = oof['image_id'].map(oof.groupby('image_id')[col].mean())
        oof = oof.drop_duplicates('image_id')
        oof = oof[oof.image_id.isin(good_image_ids)]

        scores = []
        for col in pred_cols:
            scores.append(log_loss(oof[col.replace('pred_', '')], oof[col]))
            oof = oof.rename(columns={col: f'{config}_{col}'})
            features.append(f'{config}_{col}')
        train = train.merge(oof[['image_id']+[f'{config}_{col}' for col in pred_cols]], on='image_id', how='left')
    for col in features:
        train = train[~train[col].isnull()]

    targets =['bowel_injury', 'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high']
    target_weight_map = {}
    scores = []

    y_pred = train.copy()

    print('searching config weights for each target...')

    for target in targets:
        pred_cols = [col for col in list(train) if (target in col) & (target != col)]
        predictions = train[pred_cols].values.T
        true = train[target].values
        def loss_fn(weights):
            weights = np.array(weights)/np.sum(weights)
            final_preds = 0
            for weight, pred in zip(weights, predictions):
                final_preds += weight*pred
            return log_loss(true, final_preds)

        starting_weights = [1/len(predictions)] * len(predictions)
        constraints = ({'type': 'eq', 'fun': lambda w: 1-sum(w)})
        bounds = [(0, 1)] * len(predictions)

        res = minimize(loss_fn, starting_weights, method='Nelder-Mead', bounds=bounds, constraints=constraints)
        score = res['fun']
        print(target, 'logloss:', score)
        scores.append(score)
        m = {}
        ws = res['x']
        ws = np.array(ws) / np.sum(ws)
        for w, c in zip(ws, pred_cols):
            m[c.split('_pred_')[0]] = w
        target_weight_map[target] = m
        final_preds = 0
        for weight, pred in zip(ws, predictions):
            final_preds += weight*pred
        y_pred[target] = final_preds
    print('↓target_weight_map↓')
    print(target_weight_map)
    y_pred = y_pred[['patient_id']+targets]

    y_pred['bowel_healthy'] = 1- y_pred['bowel_injury']
    y_pred['extravasation_healthy'] = 1- y_pred['extravasation_injury']


    print("searching each target weight...")
    pred_cols = ['bowel_healthy','bowel_injury','extravasation_healthy', 'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high']
    predictions = y_pred[pred_cols].values.T

    def loss_fn(weights):
        pred_df = y_pred.copy()
        for weight, col in zip(weights, pred_cols):
            pred_df[col] *= weight
        weight_scale_scores = score_(solution_train.copy(), pred_df.copy(),'patient_id')
        return np.mean(weight_scale_scores)

    starting_weights = [0.07692308, 0.19230769, 0.07692308, 0.92307692, 0.07692308,
           0.19230769, 0.46153846, 0.07692308, 0.19230769, 0.46153846,
           0.07692308, 0.19230769, 0.46153846]
    constraints = ({'type': 'eq', 'fun': lambda w: 1-sum(w)})
    bounds = [(0.00001, 0.99999)] * len(predictions)

    res = minimize(loss_fn, starting_weights, method='Nelder-Mead', bounds=bounds, constraints=constraints)
    pred_df = y_pred.copy()
    ws=res['x']
    ws = [round(w, 5) for w in ws]
    for weight, col in zip(ws, pred_cols):
        pred_df[col] *= weight
    solution_train = create_training_solution(train[['patient_id', 'any_injury','bowel_healthy','bowel_injury','extravasation_healthy', 'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high']])

    weight_scale_scores = score_(solution_train.copy(), pred_df.copy(),'patient_id')
    print('score:', round(np.mean(weight_scale_scores), 4))
    print([(m, round(v, 4)) for v,m in zip(weight_scale_scores, ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'all'])])
    print('↓column weight map↓')
    print([(c, round(w, 4)) for c, w in zip(pred_cols, ws)])
