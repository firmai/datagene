import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.regularizers import l2
import shap 
import statistics
from scipy import stats



def prepare_df_rand(arr_3d, random_state=1,frac=1):
  np.random.seed(random_state)
  arr_3d = arr_3d[np.random.choice(len(arr_3d[:,0,0]),int(len(arr_3d)*frac)),:,:]
  y = arr_3d[:,-1,-2]
  y_m1 = arr_3d[:,-2,-2]
  x = np.delete(arr_3d,-1, axis=1)
  x_tr, y_tr, x_vl, y_vl = train_test_split(x, y, test_size=0.30,random_state=random_state)
  _, y_tr_m1, _, y_vl_m1 = train_test_split(x, y_m1, test_size=0.30,random_state=random_state)

  return x_tr, y_tr, x_vl, y_vl, y_vl_m1

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

def create_time_steps(length):
  return list(range(-length, 0))
  
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def model_output(org, epochs=5,random_state=1, frac=1):

  org_x_tr, org_x_vl, org_y_tr, org_y_vl, _ = prepare_df_rand(org,random_state=random_state, frac=frac)

  BATCH_SIZE = 32
  BUFFER_SIZE = 256

  train_data_single = tf.data.Dataset.from_tensor_slices((org_x_tr, org_y_tr))
  train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  # Don't really need the validation data, reason being that it 
  val_data_single = tf.data.Dataset.from_tensor_slices((org_x_vl, org_y_vl))
  val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

  single_step_model = tf.keras.models.Sequential()
  single_step_model.add(tf.keras.layers.LSTM(16,
                                            input_shape=org_x_tr.shape[-2:],
                                            kernel_regularizer=l2(0.000001)))
  single_step_model.add(tf.keras.layers.Dropout(0.5))
  single_step_model.add(tf.keras.layers.Dense(1))

  single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

  for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)

  EVALUATION_INTERVAL = 200

  single_step_history = single_step_model.fit(train_data_single, epochs=epochs,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_single,
                                              validation_steps=50)

  plot_train_history(single_step_history,
                    'Single Step Training and validation loss')


  for x, y in val_data_single.take(3):
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                      single_step_model.predict(x)[0]], 1,
                    'Single Step Prediction')
    plot.show()

  y_pred_org = single_step_model.predict(org_x_vl)

  met = mean_absolute_error(org_y_vl, y_pred_org); print(met)

  return single_step_model, y_pred_org


def model_output_cross(org, gen, epochs=5,random_state=1, frac=1):

  org_x_tr, org_x_vl, org_y_tr, org_y_vl, org_y_vl_m1 = prepare_df_rand(org,random_state=random_state, frac=frac)
  gen_x_tr, org_x_vl, gen_y_tr, gen_y_vl, gen_y_vl_m1 = prepare_df_rand(org,random_state=random_state, frac=frac)


  BATCH_SIZE = 32
  BUFFER_SIZE = 256

  train_data_single = tf.data.Dataset.from_tensor_slices((gen_x_tr, gen_y_tr))
  train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  # Don't really need the validation data, reason being that it 
  val_data_single = tf.data.Dataset.from_tensor_slices((org_x_vl, org_y_vl))
  val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

  single_step_model = tf.keras.models.Sequential()
  single_step_model.add(tf.keras.layers.LSTM(16,
                                            input_shape=gen_x_tr.shape[-2:],
                                            kernel_regularizer=l2(0.000001)))
  single_step_model.add(tf.keras.layers.Dropout(0.5))
  single_step_model.add(tf.keras.layers.Dense(1))

  single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

  for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)

  EVALUATION_INTERVAL = 200

  single_step_history = single_step_model.fit(train_data_single, epochs=epochs,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_single,
                                              validation_steps=50)

  plot_train_history(single_step_history,
                    'Single Step Training and validation loss')


  for x, y in val_data_single.take(3):
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                      single_step_model.predict(x)[0]], 1,
                    'Single Step Prediction')
    plot.show()

  y_pred_org = single_step_model.predict(org_x_vl)

  met = mean_absolute_error(org_y_vl, y_pred_org); print(met)

  return single_step_model, y_pred_org, org_y_vl, org_y_vl_m1

#run this code for sanity checks
#org_model, org_pred = model_output(org,10)
#gen_model, gen_pred = model_output(gen, 10)


def test_shap(org,gen,rand,frac):
  org_x_tr, org_x_vl, org_y_tr, org_y_vl,_ = prepare_df_rand(org,random_state=rand)
  gen_x_tr, gen_x_vl, gen_y_tr, gen_y_vl,_ = prepare_df_rand(gen,random_state=rand+1)

  epochs = 10
  
  org_model, org_pred = model_output(org, epochs, rand+1, frac=frac)
  explainer_org = shap.GradientExplainer(org_model,org_x_vl)  
  shap_values_org = explainer_org.shap_values(org_x_vl[:int(len(org_x_vl))])

  gen_model, gen_pred = model_output(gen, epochs, rand, frac=frac) # there is little improvement by adjusting this fraction
  explainer_gen = shap.GradientExplainer(gen_model,org_x_vl) #this has to be the same otherwise problems
  shap_values_gen = explainer_gen.shap_values(org_x_vl[:int(len(org_x_vl))])  # there is no improvement by adjusting this fraction

  return shap_values_org, shap_values_gen

def shapley_rank(org, gen,f_names,frac=1,itter=10):
  arr_mean = []
  np.random.seed(0)
  for ra, rand in enumerate(np.random.choice(3000,itter)): #this list remains unchanged (seeded)
    shap_values_org, shap_values_gen = test_shap(org,gen,rand,frac)
    df_org_shap = pd.DataFrame(np.mean(np.abs(np.sum(shap_values_org[0],axis=0)),axis=0), index=f_names)
    df_gen_shap = pd.DataFrame(np.mean(np.abs(np.sum(shap_values_gen[0],axis=0)),axis=0), index=f_names)
    rank_df = pd.merge(df_org_shap, df_gen_shap, left_index=True, right_index=True, how="left")
    val = rank_df.head(10).corr(method="spearman").values[1,0]; print(val) #never select more than 10 features
    arr_mean.append(val)

    single_org_df = pd.DataFrame(np.sum(shap_values_org[0],axis=0), columns=f_names)
    single_gen_df = pd.DataFrame(np.sum(shap_values_gen[0],axis=0), columns=f_names)
    diverge_50_perc = pd.DataFrame(np.where(single_org_df>single_gen_df, 1, 0), columns=f_names)

    if ra ==0:
      divergence_total = diverge_50_perc
      single_org_total = single_org_df
      single_gen_total = single_gen_df
    else:
      divergence_total += diverge_50_perc
      single_org_total += single_org_df
      single_gen_total += single_gen_df
    print("loop {}".format(ra))

  return arr_mean, divergence_total/itter, single_org_total/itter, single_gen_total/itter


def explore_shap(org, model, cols, frac=0.11):
  explainer_org = shap.GradientExplainer(model,org)  
  shap_values_org = explainer_org.shap_values(org[:int(len(org)*frac),:,:])   

  shap_abs_org = np.absolute(shap_values_org[0])
  sum_0_org = np.sum(shap_abs_org,axis=0)

  x_pos = [i for i, _ in enumerate(cols)]
  plt1 = plt.subplot(311)
  plt1.barh(x_pos,sum_0_org[-1])
  plt1.set_yticks(x_pos)
  plt1.set_yticklabels(cols)
  plt1.set_title("Yesterday’s features (time-step -1)")
  plt2 = plt.subplot(312,sharex=plt1)
  plt2.barh(x_pos,sum_0_org[-2])
  plt2.set_yticks(x_pos)
  plt2.set_yticklabels(cols)
  plt2.set_title("The day before yesterday’s features(time-step -2)")
  plt.tight_layout()
  plt.show()
  shap.summary_plot(np.sum(shap_values_org[0],axis=0), cols, plot_type='bar', color='royalblue')


