import pandas as pd 
import numpy as np 
from collections import Counter
import re
from optbinning import OptimalBinning
from pandas.api.types import is_string_dtype



class Node: 
    def __init__(
        self, 
        Y: list,
        X: pd.DataFrame,
        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None,
        stop=None,
        dataframe_constraints= pd.DataFrame([[0,0,0]]).rename(columns={0:"path",1:"feature",2:"type_feature"}),
        path=''
    ): 
        self.Y = Y 
        self.X = X
        
        self.min_samples_split = min_samples_split if min_samples_split else 500
        self.max_depth = max_depth if max_depth else 5

        self.depth = depth if depth else 0

        self.features = list(self.X.columns)
        self.features_types = self.X.dtypes.apply(lambda x : "categorical" if is_string_dtype(x)  else "numerical").tolist()
        self.list_features = list(zip(self.features, self.features_types))

        self.node_type = node_type if node_type else 'root'

        self.rule = rule if rule else ""

        self.counts = Counter(Y)

        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        yhat = None
        y_prob = None 
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
            y_prob = np.mean(Y)

        self.yhat = yhat
        self.y_prob = y_prob

        self.n = len(Y)

        self.left = None 
        self.right = None 
        self.best_feature = None 
        self.best_value = None
        self.best_type = None
        
        self.stop = stop
        self.dataframe_constraints = dataframe_constraints
        self.path = path + str(self.depth)+self.node_type
        
    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y


        # Default best feature and split
        best_feature = None
        best_value = None
        best_type = None 
        number_level = self.depth
        list_scores = list()
        dataframe_constraints = self.dataframe_constraints[self.dataframe_constraints.feature.isin(self.features)]
        
        echantillon = dataframe_constraints[(dataframe_constraints.path == self.path)]
        if len(echantillon)>0:
            best_feature = echantillon.feature.values[0]
            feature_type = echantillon.type_feature.values[0]
            x = self.X[best_feature].values
            y = self.Y
            optb = OptimalBinning(name=best_feature, dtype=feature_type, solver="cp",max_n_bins = 2, 
                                  min_bin_size = min(0.5, self.stop / len(df)))
            optb.fit(x, y)
            binning_table = optb.binning_table.build()
            binning_table = binning_table[~binning_table["Bin"].isin( ["Special","Missing"])]
            if len(binning_table)>2:
                list_scores.append([number_level,best_feature,feature_type, 
                                    binning_table.loc[0, "Bin"], 
                                    binning_table.loc[1, "Bin"],
                                    binning_table.loc[1, "Count"],
                                    binning_table.loc[0, "Count"],
                                    binning_table.loc[0, "Event rate"], 
                                    binning_table.loc[1, "Event rate"]])
        else:       
            for feature, feature_type in self.list_features :
                x = self.X[feature].values
                y = self.Y
                optb = OptimalBinning(name=feature, dtype=feature_type, solver="cp",max_n_bins = 2, 
                                      min_bin_size = min(0.5, self.stop / len(df)))
                optb.fit(x, y)
                binning_table = optb.binning_table.build()
                binning_table = binning_table[~binning_table["Bin"].isin( ["Special","Missing"])]
                if len(binning_table)>2:
                    list_scores.append([number_level,feature,feature_type, 
                                        binning_table.loc[0, "Bin"], 
                                        binning_table.loc[1, "Bin"],
                                        binning_table.loc[1, "Count"],
                                        binning_table.loc[0, "Count"],
                                        binning_table.loc[0, "Event rate"], 
                                        binning_table.loc[1, "Event rate"]])
                
        dataframe_scores = pd.DataFrame(list_scores)
        if len(dataframe_scores) >0 :
            dataframe_scores = dataframe_scores.rename(columns={0:"level", 1:"feature", 2: "feature_type", 3:"bin_0",
                                                                4: "bin_1", 5:"count_0", 6: "count_1", 7:"event_rate_0", 8:"event_rate_1"})
            dataframe_scores['diff_rate'] = np.abs(dataframe_scores['event_rate_0']- dataframe_scores['event_rate_1'])
            max_dataframe_scores = dataframe_scores.groupby(['level'])['diff_rate'].max().reset_index().merge(dataframe_scores, how="left", on=['level', 'diff_rate'])
            best_feature = max_dataframe_scores['feature'].values[0]
            best_type = max_dataframe_scores['feature_type'].values[0]

            if best_type == "categorical":
                best_value = max_dataframe_scores['bin_0'].values[0]
            else:
                best_value = float(re.findall("\d+\.\d+", max_dataframe_scores["bin_0"].values[0])[0])

        return (best_feature, best_value, best_type)
    
    def grow_tree(self):
        df = self.X.copy()
        df['Y'] = self.Y
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            best_feature, best_value, best_type = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value
                self.best_type = best_type
                print(self.depth, best_feature,best_value, best_type )
                if self.best_type=="categorical":
                    left_df, right_df = df[df[best_feature].isin(best_value)].copy(), df[~df[best_feature].isin(best_value)].copy()
                    left = Node(left_df['Y'].values.tolist(), 
                                left_df[self.features], 
                                depth=self.depth + 1, 
                                max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split, 
                                node_type='left_node',
                                rule =" {best_feature} in {best_value} ".format(best_feature=best_feature, best_value = best_value),
                                stop = self.stop,
                                dataframe_constraints = self.dataframe_constraints,
                                path=self.path)
                    
                    self.left = left 
                    self.left.grow_tree()
                    
                    right = Node(
                        right_df['Y'].values.tolist(), 
                        right_df[self.features], 
                        depth=self.depth + 1, 
                        max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split,
                        node_type='right_node',
                        rule =" {best_feature} not in {best_value} ".format(best_feature=best_feature, best_value = best_value),
                        stop = self.stop,
                        dataframe_constraints = self.dataframe_constraints, 
                        path=self.path
                    )
                    
                    self.right = right
                    self.right.grow_tree()

                else:
                    left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()
                    left = Node(left_df['Y'].values.tolist(), 
                                left_df[self.features], 
                                depth=self.depth + 1, 
                                max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split, 
                                node_type='left_node',
                                rule = f"{best_feature} <= {round(best_value, 3)}",
                                stop = self.stop, 
                                dataframe_constraints = self.dataframe_constraints, 
                                path=self.path)
                    
                    self.left = left 
                    self.left.grow_tree()
                    
                    right = Node(
                        right_df['Y'].values.tolist(), 
                        right_df[self.features], 
                        depth=self.depth + 1, 
                        max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split,
                        node_type='right_node',
                        rule =f"{best_feature}> {round(best_value, 3)}",
                        stop = self.stop,  
                        dataframe_constraints = self.dataframe_constraints, 
                        path=self.path)
                    
                    self.right = right
                    self.right.grow_tree()
                    
        
        
    def print_tree(self, result=None):
        
        if result is None:
            result = []
 
        dictionnaire = dict(self.counts)
        result.append([self.path, self.rule,dictionnaire, round(self.y_prob, 3), self.yhat])
        if self.left is not None: 
            self.left.print_tree(result=result)
        if self.right is not None:
            self.right.print_tree(result=result)
        return result
    
    
    def print_tree_dataframe(self):
        dataframe = pd.DataFrame(self.print_tree())
        dataframe = dataframe.rename(columns={0:"path",1:"rule",2:"distribution",3:"probabilite" ,4:"prediction"})
        return dataframe


    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
            
        
            predictions.append(self.predict_obs(values))
            
        predictions = pd.DataFrame(predictions)
        predictions = predictions.rename(columns={0:"prediction",1:"probability"})
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while (cur_node.depth < cur_node.max_depth) and (cur_node.best_feature is not None):
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            type_feature = cur_node.best_type

            if cur_node.n < cur_node.min_samples_split:
                break 
                
            
            if type_feature == "categorical":
                if (values.get(best_feature) in best_value ):
                    if self.left is not None:
                        cur_node = cur_node.left
                else:
                    if self.right is not None:
                        cur_node = cur_node.right
            else:
                if (values.get(best_feature) < best_value):
                    if self.left is not None:
                        cur_node = cur_node.left
                else:
                    if self.right is not None:
                        cur_node = cur_node.right
            
        return cur_node.yhat, round(cur_node.y_prob,3)