import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier

train1 = pd.read_csv('Training 1.csv')
train2 = pd.read_csv('Training 2.csv')
train3 = pd.read_csv('Training 3.csv')
train4 = pd.read_csv('Training 4.csv')

train1["Like"], train2["Like"], train3["Like"], train4["Like"] = [0, 0, 1, 1]
train1['Best'], train2['Best'], train3['Best'], train4['Best'] = [0, 0, 0, 1]

full_info = pd.concat([train1, train2, train3, train4], ignore_index=True)

del full_info["Unnamed: 0"]

full_info['gen_simple'] = 'Na'

# Simplify genres for algorithm
for i in range(0, len(full_info)):
    if re.match('.*dance.*', full_info.loc[i, 'genre']):
        full_info.loc[i, 'gen_simple'] = 'dance'
    if re.match('.*trap.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'trap'
    if re.match('.*rap.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'rap'
    if re.match('.*indie.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'indie'
    if re.match('.*rock.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'rock'
    if re.match('.*hip hop.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'hip hop'
    if re.match('.*pop.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'pop'
    if re.match('.*country.*', full_info.loc[i, 'genre']) and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'country'
    if full_info.loc[i, 'genre'] == 'alternative metal':
        full_info.loc[i, 'gen_simple'] = 'metal'
    if full_info.loc[i, 'genre'] == 'electro swing':
        full_info.loc[i, 'gen_simple'] = 'electro swing'
    if full_info.loc[i, 'genre'] == 'edm' or full_info.loc[i, 'genre'] == 'big room':
        full_info.loc[i, 'gen_simple'] = 'edm'
    if full_info.loc[i, 'genre'] == 'boogie-woogie':
        full_info.loc[i, 'gen_simple'] = 'boogie-woogie'
    if full_info.loc[i, 'genre'] == 'latin':
        full_info.loc[i, 'gen_simple'] = 'latin'
    if full_info.loc[i, 'genre'] != 'Na' and full_info.loc[i, 'gen_simple'] == 'Na':
        full_info.loc[i, 'gen_simple'] = 'other'

# Convert genres to categories and delete non-predictive columns
full_info['gen_simple'] = full_info['gen_simple'].astype('category')
full_train = full_info.copy()
full_train.index = full_info['track_name']
del full_train['id']
del full_train['artist_id']
del full_train['artist_name']
del full_train['track_name']
del full_train['genre']

# One-hot encode categorical variables
full_train = pd.get_dummies(full_train)

fine_tuning = full_train.loc[full_train['Like'] == 1, :]
del full_train['Best']
del fine_tuning['Like']


# Create cross-validation function
def cross_valid(data, model, folds, label):
    # Sample positive and negative labels for class balance
    data_pos = data[data[label] == 1]
    data_neg = data[data[label] == 0]
    num_each = min(data_pos.shape[0], data_neg.shape[0])
    neg_frac, pos_frac = [num_each / data_neg.shape[0], num_each / data_pos.shape[0]]
    neg_sample = data_neg.sample(frac=neg_frac)
    pos_sample = data_pos.sample(frac=pos_frac)

    # Merge and shuffle data
    shuffled_data = pd.concat([neg_sample, pos_sample]).copy().sample(frac=1)
    shuffled_data['fold'] = -1
    fold_size = len(shuffled_data) / folds

    # Assign fold for cross-validation
    for i in range(folds):
        shuffled_data['fold'][int(i * fold_size):int((i + 1) * fold_size)] = i
    for i in range(folds):
        test_data = shuffled_data.loc[shuffled_data['fold'] == i, :]
        train_data = shuffled_data.loc[shuffled_data['fold'] != i, :]
        x_test, y_test = test_data.drop(label, axis=1), test_data.loc[:, label]
        x_train, y_train = train_data.drop(label, axis=1), train_data.loc[:, label]
        model.fit(x_train, y_train)
        predicted_val = model.predict(x_test)
        accuracy = sum(predicted_val == y_test) / len(y_test)

        print(accuracy)


rf_model = RandomForestClassifier(n_estimators=50, max_features='sqrt')
cross_valid(fine_tuning, rf_model, 5, 'Best')
cross_valid(full_train, rf_model, 5, 'Like')

