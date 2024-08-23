import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import Union, Tuple
from statsmodels.api import GLM
from sklearn import metrics

class Esmlp:
    """

    Intro
    ----------------------------
    This class provides apis to process the esmlp method in link -predictions.

    You can use the 'matrix_divide' function to divide raw datas into training sets and test sets.

    API
    ----------------------------

    Esmlp.matrix_adjacency(node_num: int, ori_data: np.array) -> np.array
    This function provides a transformation from a node relation to a spatial adjacency matrix.

    Esmlp.matrix_divide(ori_data: np.array,
                  matrix_data: np.array,
                  node_num: int,
                  train_per: float = 0.8) -> Tuple[np.array, np.array]
    This function provides a division of training sets and test sets.

    instance.process(self) -> None:
    This function provides an api to run the process.

    OutPut
    ----------------------------
    Prints(default) or csv files(if you choose).

    Author
    ----------------------------
    ZQH、CXY

    """

    # ========InnerClass========

    def __new__(cls,
                train_matrix:np.array,
                test_matrix: np.array,
                node_num: int,
                eco_matrix: np.array = None,
                parameter: Union[float, None] = None,
                write: bool = False):
        """
        New function.
        :param train_matrix: train sets.
        :param test_matrix: test sets.
        :param node_num: nodes counts.
        :param eco_matrix: Prior(economic) knowledge.
        :param parameter: parameter in link predictions
        :param write: whether to write results into csv files.
        """
        if train_matrix.ndim <= 1:
            raise ValueError(f'The training set dimension={train_matrix.ndim} is incorrect.')
        if eco_matrix is not None:
            if eco_matrix.ndim <= 1:
                raise ValueError(f'The economic weight matrix dimension={eco_matrix.ndim} is incorrectly correct.')
        if test_matrix.ndim <= 1:
            raise ValueError(f'The test set dimension={test_matrix.ndim} is incorrectly correct.')
        if int(train_matrix.shape[0]) != int(test_matrix.shape[0]) or int(train_matrix.shape[1]) != int(test_matrix.shape[1]):
            raise ValueError(f'The {train_matrix.shape} and {test_matrix.shape} dimensions of the training set are different from those of the test set, reclassify the training set and the test set.')
        rows: int = len(train_matrix)
        if rows != node_num:
            raise ValueError(f'The number of nodes entered = {node_num} does not match the dimension of the training set = {rows}. ')
        for row in train_matrix:
            if len(row) != rows:
                raise AttributeError(f'The training set is not a square, try using the Esmlp.matrix_adjacency method to convert it into a square.')
        if eco_matrix is not None:
            rows: int = len(eco_matrix)
            if rows != node_num:
                raise ValueError(f'The number of nodes entered = {node_num} does not correspond to the dimension of a priori economic data = {rows}.')
            for row in eco_matrix:
                if len(row) != rows:
                    raise AttributeError(f'The economic weight matrix matrix is not a square.')
        rows: int = len(test_matrix)
        for row in test_matrix:
            if len(row) != rows:
                raise AttributeError(f'The test set is not a square.')
        instance: Esmlp = super().__new__(cls)
        return instance

    def __init__(self,
                 train_matrix: np.array,
                 test_matrix: np.array,
                 node_num: int,
                 eco_matrix: np.array = None,
                 parameter: Union[float, None] = None,
                 write: bool = False) -> None:
        """
        Init function.
        :param train_matrix: train sets.
        :param test_matrix: test sets.
        :param node_num: nodes counts.
        :param eco_matrix: Prior(economic) knowledge.
        :param parameter: parameter in link predictions
        :param write: whether to write results into csv files.
        """
        self.train_matrix: np.array = train_matrix
        self.test_matrix: np.array = test_matrix
        self.node_num: int = node_num
        self.eco_matrix: np.array = eco_matrix
        self.parameter: Union[float, None] = parameter
        self.write: bool = write

    @staticmethod
    def __sparse(array: np.array) -> np.array:
        for index in range(len(array)):
            if array[index] != 0:
                array[index] = 1
        return array

    @staticmethod
    def matrix_adjacency(ori_data: np.array) -> np.array:
        # This function provides a transformation from a node relation to a spatial adjacency matrix.
        """
        :param ori_data: node relations.
        """
        list_one: list = []
        list_two: list = []
        for row in range(ori_data.shape[0]):
            list_one.append(ori_data[row][0])
            list_two.append(ori_data[row][1])
        list_one: list = list(set(list_one))
        list_two: list = list(set(list_two))
        node_num: int = int(max(max(list_one), max(list_two))) + 1
        matrix_adjacency: np.array = np.zeros([node_num, node_num])
        if len(ori_data[0]) != 2 or len(ori_data[1]) != 2:
            raise AttributeError(f'The column dimension of the original space node relationship is incorrectly {len(ori_data[0]), len(ori_data[1])}, note that the dimension is 2.')
        for column in range(ori_data.shape[0]):
            i: int = int(ori_data[column][0])
            j: int = int(ori_data[column][1])
            matrix_adjacency[i, j] = 1
            matrix_adjacency[j, i] = 1
        return matrix_adjacency

    @ staticmethod
    def matrix_divide(ori_data: np.array,
                      matrix_data: np.array,
                      node_num: int,
                      train_per: float = 0.8) -> Tuple[np.array, np.array]:
        # This function provides a division of training sets and test sets.
        """
        :param ori_data: node relations.
        :param matrix_data: spatial adjacency matrix.
        :param node_num: node counts.
        :param train_per: percent that train sets compare to raw datas.
        """
        train_per: float = train_per
        ori_data: np.array = ori_data
        num_data: int = ori_data.shape[0]
        num_test: int = int(float(1 - train_per) * num_data)
        node_num: int = node_num + 1
        test_matrix: np.array = np.zeros([node_num, node_num])
        while len(np.nonzero(test_matrix)[0]) < num_test:
            link_index: int = int(np.random.rand(1) * ori_data.shape[0])
            id_one: int = int(ori_data[link_index, 0])
            id_two: int = int(ori_data[link_index, 1])
            matrix_data[id_one, id_two] = 0
            matrix_data[id_two, id_one] = 0
            temp: np.array = matrix_data[id_one]
            sign: int = 0
            one_to_two: np.array = np.dot(temp, matrix_data) + temp
            if one_to_two[id_two] > 0:
                sign: int = 1
            else:
                count: int = 1
                while len((Esmlp.__sparse(one_to_two) - temp).nonzero()[0]) != 0:
                    temp: np.array = Esmlp.__sparse(one_to_two)
                    one_to_two: np.array = np.dot(temp, matrix_data) + temp
                    count: int = count + 1
                    if one_to_two[id_two] > 0:
                        sign: int = 1
                        break
                    if count >= matrix_data.shape[0]:
                        sign: int = 0
            if sign == 1:
                ori_data: np.array = np.delete(ori_data, link_index, axis=0)
                test_matrix[id_one, id_two] = 1
            else:
                ori_data = np.delete(ori_data, link_index, axis=0)
                matrix_data[id_one, id_two] = 1
                matrix_data[id_two, id_one] = 1
        train_matrix: np.array = matrix_data
        test_matrix: np.array = test_matrix + test_matrix.T
        return train_matrix, test_matrix

    def __calauc(self, similarity_matrix: np.array) -> float:
        # This function is to calculate AUC from results.
        """
        :param similarity_matrix: similarity_matrix.
        """
        simi_ravel: np.array = np.ravel(similarity_matrix)
        test_ravel: np.array = np.ravel(self.test_matrix)
        for index in range(len(simi_ravel)):
            if np.isnan(simi_ravel[index]):
                simi_ravel[index] = 0
        fpr, tpr, thresholds = metrics.roc_curve(y_true=test_ravel, y_score=simi_ravel)
        auc: float = metrics.auc(fpr,tpr)
        return auc

    def __get_parameters(self) -> float:
        # This function is to calculate parameters using prior(economic) knowledge.
        model: GLM = sm.GLM(np.ravel(self.eco_matrix), np.ravel(self.train_matrix @ self.eco_matrix))
        res_glm = model.fit()
        estimated_parameters: np.array = res_glm.params
        if float(np.mean(res_glm.df_resid)) >= 0:
            estimated_parameter: float = float(np.mean(estimated_parameters))
            return estimated_parameter
        else:
            raise AttributeError(f'The parameters of the existing economic indicator fail to fit, please find another economic indicator as the prior knowledge(thresholds).')

    # ========Methods========

    def __katz(self) -> Tuple[np.array, float]:
        """
        Katz Method.
        """
        # katz_parameter is changeable
        katz_parameter: float = self.parameter if self.parameter is not None else 0.01
        identity_matrix: np.array = np.eye(self.train_matrix.shape[0])
        temp: np.array = self.train_matrix - self.train_matrix * katz_parameter
        similarity_matrix: np.array = np.linalg.pinv(temp)
        similarity_matrix: np.array = similarity_matrix - identity_matrix
        return similarity_matrix, katz_parameter

    def __lnh(self) -> Tuple[np.array, str]:
        """
        LNH-I Method.
        """
        # no parameters for such method
        similarity_matrix: np.array = np.dot(self.train_matrix, self.train_matrix)
        deg_row = sum(self.train_matrix)
        deg_row.shape = (deg_row.shape[0], 1)
        deg_row_t: np.array = deg_row.T
        temp: np.array = np.dot(deg_row, deg_row_t)
        similarity_matrix:np.array = similarity_matrix / temp
        return similarity_matrix, 'NaN'

    def __lp(self) -> Tuple[np.array, float]:
        """
        LP Method.
        """
        # lp_parameter is changeable
        lp_parameter: float = self.parameter if self.parameter is not None else 1
        similarity_matrix: np.array = np.dot(self.train_matrix, self.train_matrix)
        temp: np.array = np.dot(np.dot(self.train_matrix, self.train_matrix), self.train_matrix) * lp_parameter
        similarity_matrix: np.array = np.dot(similarity_matrix, temp)
        return similarity_matrix, lp_parameter

    def __sorenson(self) -> Tuple[np.array, str]:
        """
        Sorenson Method.
        """
        # no parameters for such method
        similarity_matrix: np.array = np.dot(self.train_matrix, self.train_matrix)
        deg_row = sum(self.train_matrix)
        deg_row.shape = (deg_row.shape[0], 1)
        deg_row_t = deg_row.T
        temp: np.array = deg_row + deg_row_t
        similarity_matrix: np.array = (2 * similarity_matrix) / temp
        return similarity_matrix, 'NaN'

    def __ra(self) -> Tuple[np.array, str]:
        """
        RA Method.
        """
        # no parameters for such method
        ra_train = sum(self.train_matrix)
        ra_train.shape = (ra_train.shape[0], 1)
        temp_train_log: np.array = self.train_matrix / ra_train
        temp_train_log: np.array = np.nan_to_num(temp_train_log)
        similarity_matrix: np.array = np.dot(self.train_matrix, temp_train_log)
        return similarity_matrix, 'NaN'

    def __esmlp(self) -> Tuple[np.array, float]:
        """
        ESMLP Method.
        """
        if self.parameter is not None and self.eco_matrix is not None:
            raise AttributeError(f'The prior parameter = {self.parameter} is still declared under ESMLP method.')
        elif self.parameter is not None and self.eco_matrix is None:
            esmlp_parameter: float = self.parameter
        elif self.parameter is None and self.eco_matrix is not None:
            esmlp_parameter: float = self.__get_parameters()
        else:
            raise AttributeError('The ESMLP method does not give a priori parameters without declaring economic indicators.')
        temp: np.array = esmlp_parameter * self.train_matrix
        for ind in range(len(temp)):
            for jnd in range(len(temp[0])):
                temp[ind][jnd] = np.exp(temp[ind][jnd]) - 1
                temp[ind][jnd] = 0 if temp[ind][jnd] < 0 else temp[ind][jnd]
        similarity_matrix: np.array = temp
        return similarity_matrix, esmlp_parameter

    # ========Process========

    def process(self) -> None:
        """
        This function provides an api to run the process.
        """
        pre_result: dict = {'Method':['Katz','LNH-I','LP','Sorenson','RA','ESMLP'],
                            'AUC':[],
                            'Parameter':[]}
        print('=======Info=======')
        print('Info：keep four decimal places.')
        if self.eco_matrix is not None:
            print('Info：prior knowledge used.')
        if self.parameter is not None:
            print(f'Info: parameter={self.parameter}')
        print('=======Info=======')
        print('processing...')
        # Katz
        simi_matrix_katz, parameter_katz = self.__katz()
        auc_katz: float = self.__calauc(similarity_matrix=simi_matrix_katz)
        pre_result['AUC'].append(format(auc_katz,'.4f'))
        pre_result['Parameter'].append(format(parameter_katz,'.4f'))
        # LNH-I
        simi_matrix_lnh, parameter_lnh = self.__lnh()
        auc_lnh: float = self.__calauc(similarity_matrix=simi_matrix_lnh)
        pre_result['AUC'].append(format(auc_lnh,'.4f'))
        pre_result['Parameter'].append(parameter_lnh)
        # LP
        simi_matrix_lp, parameter_lp = self.__lp()
        auc_lp: float = self.__calauc(similarity_matrix=simi_matrix_lp)
        pre_result['AUC'].append(format(auc_lp,'.4f'))
        pre_result['Parameter'].append(format(parameter_lp,'.4f'))
        # Sorenson
        simi_matrix_so, parameter_so = self.__sorenson()
        auc_so: float = self.__calauc(similarity_matrix=simi_matrix_so)
        pre_result['AUC'].append(format(auc_so,'.4f'))
        pre_result['Parameter'].append(parameter_so)
        # RA
        simi_matrix_ra, parameter_ra = self.__ra()
        auc_ra: float = self.__calauc(similarity_matrix=simi_matrix_ra)
        pre_result['AUC'].append(format(auc_ra,'.4f'))
        pre_result['Parameter'].append(parameter_ra)
        # ESMLP
        simi_matrix_es, parameter_es = self.__esmlp()
        auc_es: float = self.__calauc(similarity_matrix=simi_matrix_es)
        pre_result['AUC'].append(format(auc_es,'.4f'))
        pre_result['Parameter'].append(format(parameter_es,'.4f'))
        print('=======Result=======')
        print(f'{pre_result}')
        print('=======Result=======')
        if self.write is False:
            print('No write.')
        else:
            data_esmlp: DataFrame = pd.DataFrame(simi_matrix_es)
            data_esmlp.to_csv(f'ESMLP_RESULT_PAR={self.parameter}.csv')
            print(f'Write results into ESMLP_RESULT_PAR={self.parameter}.csv, completed')

# ========Demo========
if __name__ == '__main__':
    # COLoil
    '''
    train_set: np.array = (pd.read_csv('COLoil_Train.csv')).values
    eco_set: np.array = (pd.read_csv('COLoil_Eco.csv')).values
    test_set: np.array = (pd.read_csv('COLoil_Test.csv')).values
    # prediction_one to get the parameter of ESMLP(using prior economic knowledge)
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=24, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=24, parameter=0.2290)
    prediction_one.process()
    prediction_two.process()
    '''
    # GZRway
    '''
    train_set: np.array = (pd.read_csv('GZRway_Train.csv')).values
    eco_set: np.array = (pd.read_csv('GZRway_Eco.csv')).values
    test_set: np.array = (pd.read_csv('GZRway_Test.csv')).values
    # prediction_one to get the parameter of ESMLP(using prior economic knowledge)
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=23, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=23, parameter=0.1980)
    prediction_one.process()
    prediction_two.process()
    '''
    # HNRway
    '''
    train_set: np.array = (pd.read_csv('HNRway_Train.csv')).values
    eco_set: np.array = (pd.read_csv('HNRway_Eco.csv')).values
    test_set: np.array = (pd.read_csv('HNRway_Test.csv')).values
    # prediction_one to get the parameter of ESMLP(using prior economic knowledge)
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=17, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=17, parameter=0.1909)
    prediction_one.process()
    prediction_two.process()
    '''
    # CSJRway
    '''
    train_set: np.array = (pd.read_csv('CSJRway_Train.csv')).values
    eco_set: np.array = (pd.read_csv('CSJRway_Eco.csv')).values
    test_set: np.array = (pd.read_csv('CSJRway_Test.csv')).values
    # prediction_one to get the parameter of ESMLP(using prior economic knowledge)
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=26, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=26, parameter=0.1690)
    prediction_one.process()
    prediction_two.process()
    '''
    # USAir
    '''
    # data pre-processing
    ori_set: np.array = (pd.read_csv('USAir.csv')).values
    ori_data_matrix: np.array = Esmlp.matrix_adjacency(ori_data=ori_set)
    train_data, test_data = Esmlp.matrix_divide(ori_data=ori_set, matrix_data=ori_data_matrix, node_num=332)
    pd.DataFrame(train_data).to_csv(f'USAir_Train.csv')
    pd.DataFrame(test_data).to_csv(f'USAir_Test.csv')
    '''
    '''
    train_set: np.array = (pd.read_csv('USAir_Train.csv')).values
    test_set: np.array = (pd.read_csv('USAir_Test.csv')).values
    # prediction_one to get the parameter of ESMLP(using prior economic knowledge)
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=332, parameter=0.1)
    prediction_one.process()
    '''
    # Plot USAir
    '''
    pred_data = pd.read_csv('draw.csv')
    x_par = np.array(pred_data['Par'])
    y_katz = np.array(pred_data['Katz'])
    y_lnh = np.array(pred_data['LNH-I'])
    y_lp = np.array(pred_data['LP'])
    y_so = np.array(pred_data['Sorenson'])
    y_ra = np.array(pred_data['RA'])
    y_es = np.array(pred_data['ESMLP'])
    color_list = plt.cm.plasma(np.linspace(0, 1, 30))
    plt.figure(1)
    plt.title('Link Prediction Comparative Analysis')
    plt.xlabel('Parameter')
    plt.ylabel('AUC')
    plt.plot(x_par, y_katz, color=color_list[26], marker='o')
    plt.plot(x_par, y_lnh, color=color_list[22], marker='o')
    plt.plot(x_par, y_lp, color=color_list[18], marker='o')
    plt.plot(x_par, y_so, color=color_list[14], marker='*')
    plt.plot(x_par, y_ra, color=color_list[10], marker='*')
    plt.plot(x_par, y_es, color=color_list[6], marker='o')
    plt.legend(labels=['Katz','LNH-I','LP','Sorenson','RA','ESMLP'], loc='best')
    plt.show()
    '''
