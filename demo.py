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

    介绍
    ----------------------------
    训练集和测试集可以使用matrix_adjacency自动划分。

    调用接口
    ----------------------------

    Esmlp.matrix_adjacency(node_num: int, ori_data: np.array) -> np.array
    该函数提供了由节点关系到空间邻接矩阵的转换。
    
    Esmlp.matrix_divide(ori_data: np.array,
                  matrix_data: np.array,
                  node_num: int,
                  train_per: float = 0.8) -> Tuple[np.array, np.array]
    该函数提供了训练集与测试集的划分。

    instance.process(self) -> None:
    该函数用于执行链路预测过程。

    输出
    ----------------------------
    输出为打表及CSV写入。

    作者
    ----------------------------
    ZQH、CXY
    详情请见ReadMe

    """

    # ========功能实现部分========

    def __new__(cls,
                train_matrix:np.array,
                test_matrix: np.array,
                node_num: int,
                eco_matrix: np.array = None,
                parameter: Union[float, None] = None,
                write: bool = False):
        """
        构造函数。
        :param train_matrix: 空间邻接矩阵（或训练集）
        :param test_matrix: 真实链路矩阵（或测试集）
        :param node_num: 节点数
        :param eco_matrix: 经济权重矩阵（如默认为None则无经济指标作为门限）
        :param parameter: 除Esmlp模型外的模型最优参数（如默认为None则使用默认参数）
        :param write: 是否将预测结果写入CSV表（如默认为False则不写入）
        """
        if train_matrix.ndim <= 1:
            raise ValueError(f'训练集维度错误为{train_matrix.ndim}。')
        if eco_matrix is not None:
            if eco_matrix.ndim <= 1:
                raise ValueError(f'经济权重矩阵维度错误为{eco_matrix.ndim}。')
        if test_matrix.ndim <= 1:
            raise ValueError(f'测试集维度错误为{test_matrix.ndim}。')
        if int(train_matrix.shape[0]) != int(test_matrix.shape[0]) or int(train_matrix.shape[1]) != int(test_matrix.shape[1]):
            raise ValueError(f'训练集{train_matrix.shape}与测试集{test_matrix.shape}维度不一致，请重新划分训练集与测试集。')
        rows: int = len(train_matrix)
        for row in train_matrix:
            if len(row) != rows:
                raise AttributeError(f'训练集不是方阵，请尝试使用Esmlp.matrix_adjacency方法转换为方阵。')
        if eco_matrix is not None:
            rows: int = len(eco_matrix)
            for row in eco_matrix:
                if len(row) != rows:
                    raise AttributeError(f'经济权重矩阵矩阵不是方阵。')
        rows: int = len(test_matrix)
        for row in test_matrix:
            if len(row) != rows:
                raise AttributeError(f'测试集不是方阵。')
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
        属性函数。
        :param train_matrix: 空间邻接矩阵（或训练集）
        :param test_matrix: 真实链路矩阵（或测试集）
        :param node_num: 节点数
        :param eco_matrix: 经济指标矩阵（如默认为None则无经济指标作为门限）
        :param parameter: 除Esmlp模型外的模型最优参数（如默认为None则使用默认参数）
        :param write: 是否将预测结果写入CSV表（如默认为False则不写入）
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
        """
        该函数提供了由节点关系到空间邻接矩阵转换。
        :param ori_data: 原始节点关系
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
            raise AttributeError(f'原始空间节点关系的列维度错误为{len(ori_data[0]), len(ori_data[1])}，请注意该维度为2。')
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
        """
        该函数提供了训练集与测试集的划分。
        :param ori_data: 原始节点关系
        :param matrix_data: 原始节点的邻接矩阵
        :param node_num: 节点数
        :param train_per: 测试集占比（此项可以自己更改，默认0.8）
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
        """
        该函数用于计算AUC值。
        :param self: 实例
        :param similarity_matrix: 预测结果的相似矩阵
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
        """
        获取具备经济意义的参数参考值。
        :param self: 实例
        """
        model: GLM = sm.GLM(np.ravel(self.eco_matrix), np.ravel(self.train_matrix @ self.eco_matrix))
        res_glm = model.fit()
        estimated_parameters: np.array = res_glm.params
        if float(np.mean(res_glm.df_resid)) >= 0:
            estimated_parameter: float = float(np.mean(estimated_parameters))
            return estimated_parameter
        else:
            raise AttributeError(f'对现有经济指标的参数拟合失败，请重新寻找经济指标作为门限。')

    # ========方法实现部分========

    def __katz(self) -> Tuple[np.array, float]:
        """
        Katz链路预测方法（基于路径）。
        :param self: 实例
        """
        # katz_parameter这一参数可以自己更改
        katz_parameter: float = self.parameter if self.parameter is not None else 0.01
        identity_matrix: np.array = np.eye(self.train_matrix.shape[0])
        temp: np.array = self.train_matrix - self.train_matrix * katz_parameter
        similarity_matrix: np.array = np.linalg.pinv(temp)
        similarity_matrix: np.array = similarity_matrix - identity_matrix
        return similarity_matrix, katz_parameter

    def __lnh(self) -> Tuple[np.array, str]:
        """
        LNH-I链路预测方法（基于路径）。
        :param self: 实例
        """
        # 该方法无先验参数
        similarity_matrix: np.array = np.dot(self.train_matrix, self.train_matrix)
        deg_row = sum(self.train_matrix)
        deg_row.shape = (deg_row.shape[0], 1)
        deg_row_t: np.array = deg_row.T
        temp: np.array = np.dot(deg_row, deg_row_t)
        similarity_matrix:np.array = similarity_matrix / temp
        return similarity_matrix, 'NaN'

    def __lp(self) -> Tuple[np.array, float]:
        """
        LP链路预测方法（Katz方法变种）。
        :param self: 实例
        """
        # lp_parameter这一参数也可以自己更改
        lp_parameter: float = self.parameter if self.parameter is not None else 1
        similarity_matrix: np.array = np.dot(self.train_matrix, self.train_matrix)
        temp: np.array = np.dot(np.dot(self.train_matrix, self.train_matrix), self.train_matrix) * lp_parameter
        similarity_matrix: np.array = np.dot(similarity_matrix, temp)
        return similarity_matrix, lp_parameter

    def __sorenson(self) -> Tuple[np.array, str]:
        """
        Sorenson链路预测方法(LNH方法变种)。
        :param self: 实例
        """
        # 该方法无先验参数
        similarity_matrix: np.array = np.dot(self.train_matrix, self.train_matrix)
        deg_row = sum(self.train_matrix)
        deg_row.shape = (deg_row.shape[0], 1)
        deg_row_t = deg_row.T
        temp: np.array = deg_row + deg_row_t
        similarity_matrix: np.array = (2 * similarity_matrix) / temp
        return similarity_matrix, 'NaN'

    def __ra(self) -> Tuple[np.array, str]:
        """
        RA链路预测方法(传统链路预测方法/基于共同邻居)。
        :param self: 实例
        """
        # 该方法无先验参数
        ra_train = sum(self.train_matrix)
        ra_train.shape = (ra_train.shape[0], 1)
        temp_train_log: np.array = self.train_matrix / ra_train
        temp_train_log: np.array = np.nan_to_num(temp_train_log)
        similarity_matrix: np.array = np.dot(self.train_matrix, temp_train_log)
        return similarity_matrix, 'NaN'

    def __esmlp(self) -> Tuple[np.array, float]:
        """
        ESMLP链路预测方法(本文方法)。
        :param self: 实例
        """
        # 该方法可有先验参数可无先验参数
        if self.parameter is not None and self.eco_matrix is not None:
            raise AttributeError(f'使用经济指标进行计算参数下仍声明了先验参数={self.parameter}。')
        elif self.parameter is not None and self.eco_matrix is None:
            esmlp_parameter: float = self.parameter
        elif self.parameter is None and self.eco_matrix is not None:
            esmlp_parameter: float = self.__get_parameters()
        else:
            raise AttributeError('ESMLP方法在未声明经济指标情况下也未给予先验参数。')
        temp: np.array = esmlp_parameter * self.train_matrix
        for ind in range(len(temp)):
            for jnd in range(len(temp[0])):
                temp[ind][jnd] = np.exp(temp[ind][jnd]) - 1
                temp[ind][jnd] = 0 if temp[ind][jnd] < 0 else temp[ind][jnd]
        similarity_matrix: np.array = temp
        return similarity_matrix, esmlp_parameter

    # ========调用实现部分========

    def process(self) -> None:
        """
        执行接口。
        :param self: 实例
        """
        pre_result: dict = {'Method':['Katz','LNH-I','LP','Sorenson','RA','ESMLP'],
                            'AUC':[],
                            'Parameter':[]}
        print('=======预测备注=======')
        print('备注：所有结果均保留四位小数')
        if self.eco_matrix is not None:
            print('备注：对ESMLP方法参数使用了经济矩阵。')
        if self.parameter is not None:
            print(f'备注：对所有方法均使用了先验参数={self.parameter}')
        print('=======预测备注=======')
        print('正在执行，请稍等...')
        # Katz方法
        simi_matrix_katz, parameter_katz = self.__katz()
        auc_katz: float = self.__calauc(similarity_matrix=simi_matrix_katz)
        pre_result['AUC'].append(format(auc_katz,'.4f'))
        pre_result['Parameter'].append(format(parameter_katz,'.4f'))
        # LNH-I方法
        simi_matrix_lnh, parameter_lnh = self.__lnh()
        auc_lnh: float = self.__calauc(similarity_matrix=simi_matrix_lnh)
        pre_result['AUC'].append(format(auc_lnh,'.4f'))
        pre_result['Parameter'].append(parameter_lnh)
        # LP方法
        simi_matrix_lp, parameter_lp = self.__lp()
        auc_lp: float = self.__calauc(similarity_matrix=simi_matrix_lp)
        pre_result['AUC'].append(format(auc_lp,'.4f'))
        pre_result['Parameter'].append(format(parameter_lp,'.4f'))
        # Sorenson方法
        simi_matrix_so, parameter_so = self.__sorenson()
        auc_so: float = self.__calauc(similarity_matrix=simi_matrix_so)
        pre_result['AUC'].append(format(auc_so,'.4f'))
        pre_result['Parameter'].append(parameter_so)
        # RA方法
        simi_matrix_ra, parameter_ra = self.__ra()
        auc_ra: float = self.__calauc(similarity_matrix=simi_matrix_ra)
        pre_result['AUC'].append(format(auc_ra,'.4f'))
        pre_result['Parameter'].append(parameter_ra)
        # ESMLP方法
        simi_matrix_es, parameter_es = self.__esmlp()
        auc_es: float = self.__calauc(similarity_matrix=simi_matrix_es)
        pre_result['AUC'].append(format(auc_es,'.4f'))
        pre_result['Parameter'].append(format(parameter_es,'.4f'))
        print('=======预测结果=======')
        print(f'{pre_result}')
        print('=======预测结果=======')
        if self.write is False:
            print('未设置将结果写入，预测流程结束。')
        else:
            data_esmlp: DataFrame = pd.DataFrame(simi_matrix_es)
            data_esmlp.to_csv(f'ESMLP_RESULT_PAR={self.parameter}.csv')
            print(f'已将结果写入ESMLP_RESULT_PAR={self.parameter}.csv，预测流程结束。')

# ========代码示范========
if __name__ == '__main__':
    # COLoil数据
    '''
    train_set: np.array = (pd.read_csv('COLoil_Train.csv')).values
    eco_set: np.array = (pd.read_csv('COLoil_Eco.csv')).values
    test_set: np.array = (pd.read_csv('COLoil_Test.csv')).values
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=24, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=24, parameter=0.2290)
    prediction_one.process()
    prediction_two.process()
    '''
    # GZRway数据
    '''
    train_set: np.array = (pd.read_csv('GZRway_Train.csv')).values
    eco_set: np.array = (pd.read_csv('GZRway_Eco.csv')).values
    test_set: np.array = (pd.read_csv('GZRway_Test.csv')).values
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=21, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=21, parameter=0.2207)
    prediction_one.process()
    prediction_two.process()
    '''
    # HNRway数据
    '''
    train_set: np.array = (pd.read_csv('HNRway_Train.csv')).values
    eco_set: np.array = (pd.read_csv('HNRway_Eco.csv')).values
    test_set: np.array = (pd.read_csv('HNRway_Test.csv')).values
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=17, eco_matrix=eco_set)
    prediction_two = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=17, parameter=0.1909)
    prediction_one.process()
    prediction_two.process()
    '''
    # USAir数据(数据处理)
    '''
    # 数据处理过程
    ori_set: np.array = (pd.read_csv('USAir.csv')).values
    ori_data_matrix: np.array = Esmlp.matrix_adjacency(ori_data=ori_set)
    train_data, test_data = Esmlp.matrix_divide(ori_data=ori_set, matrix_data=ori_data_matrix, node_num=332)
    pd.DataFrame(train_data).to_csv(f'USAir_Train.csv')
    pd.DataFrame(test_data).to_csv(f'USAir_Test.csv')
    '''
    # USAir数据(下面重新读取时记得先在excel中把数据的第一列删掉，因为pd默认是按第一行作为指标值的)
    '''
    train_set: np.array = (pd.read_csv('USAir_Train.csv')).values
    test_set: np.array = (pd.read_csv('USAir_Test.csv')).values
    prediction_one = Esmlp(train_matrix=train_set, test_matrix=test_set, node_num=332, parameter=0.1)
    prediction_one.process()
    '''
    # USAir数据(重复上面的操作得到不同参数下的结果并绘制折线图)
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




