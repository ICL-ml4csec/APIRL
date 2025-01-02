import pandas as pd
import numpy as np


class DynamicStorage:
   def __init__(self):
      self.id = []
      self.keys = []
      self.store = pd.DataFrame()


   def add_column(self, key):
      self.keys.append(key)
      self.store[key] = None

   def add_row(self, values):

      index_map = {v: i for i, v in enumerate(self.keys)}
      for key in self.keys:

         if key not in values:
            values[key] = None
         if type(values[key]) in [int, float]:
            values[key] = str(values[key])
         if type(values[key]) == dict:
            print('kn')
      #if None not in values.values() or values['status'] == '405':
      #   print('none not here')

      sorted_values = sorted(values.items(), key=lambda pair: index_map[pair[0]])
      sorted_values = {key: value for key, value in sorted_values}
      #self.store = self.store.concat(sorted_values, ignore_index=True)
      #pd.concat(self.store, sorted_values)
      self.store = pd.concat([self.store,  pd.DataFrame([sorted_values])], ignore_index=True)
      #self.store = self.store.append(sorted_values, ignore_index=True)

   def insert_value(self, insert_value, relation_value):
      id = self.get_id_from_val(relation_value)
      header = list(insert_value)[0]
      if type( insert_value[header]) == dict:
         print('fuck')
      self.store.at[id, header] = insert_value[header]

   def delete_value(self, value):
      id = self.get_id_from_val(value)
      header =  list(value)[0]
      self.store.at[id, header] = None
      if all(value == None for value in self.store.iloc[id].values):
         self.store = self.store.drop(id, axis=0)

   def in_table(self, value):
      if type(value) in [int, bool, float] or value == None or len(value) > 0:
         return value in self.store.values
      else:
         return False

   def get_id_from_val(self, value):
      #try:

      header = list(value)[0]
      return np.where(self.store[header]==value[header])[0][0]
      #except:
      #   print('kj ')

   def get_values_from_value(self, value):
      id = self.get_id_from_val(value)
      values = self.store.loc[id].to_dict()
      values = {key: val for key, val in values.items() if val != None}
      return values

   def get_related_keys(self, header):
      not_null_values = self.store[self.store[header].notnull()]
      related_keys = [key for key in not_null_values.keys() if len(not_null_values[not_null_values[key].notnull()]) > 0]
      return related_keys
