import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference


import numpy as np
from pgmpy.models import DynamicBayesianNetwork as DBN
import pickle
from classifier import Classifier


if __name__ == '__main__':
    data_path = "./jena_climate_2009_2016.csv"
    dataset = pd.read_csv(data_path, parse_dates=['Date Time'], index_col=['Date Time'])

    dataset['year'] = dataset.index.year
    dataset['month'] = dataset.index.month
    dataset['hour'] = dataset.index.hour

    num_train = 400000
    num_test = 20451

    dataset_train = dataset[:num_train]
    dataset_test = dataset[num_train:]

    scale = 2
    cluster_v = [3, 3, 3, 5, 3, 3, 4, 2, 4, 4, 3, 2, 2, 4]
    cluster_l = np.zeros(shape=len(cluster_v))
    cluster_v = cluster_v[:-1]
    cluster_v = [x * scale for x in cluster_v]

    data = dataset_train.values[:, :-4].T

    mycls = Classifier(cluster_v=cluster_v)
    labels = mycls.fit_predict(data)
    labels = np.array(labels).T

    # avg
    avg = np.mean(data, axis=1)
    print('avg: ' + str(avg))
    print('\n')

    # std
    std = np.std(data, axis=1)
    print('std: ' + str(std))
    print('\n')

    # Correlation Coefficient
    cc = np.zeros((len(avg), len(avg)))
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            cc[i, j] = (np.mean(data[i] * data[j]) - avg[i] * avg[j]) / (std[i] * std[j])

    # We use 2 days data to predict the 3th day's situation
    windows = 3
    sample = 24

    fit_data = np.zeros(shape=(labels.shape[0] - windows, labels.shape[1] * windows))
    for i in range(windows):
        fit_data[:, labels.shape[1] * i:labels.shape[1] * (i + 1)] = labels[i:-windows + i, :]
    fit_data = np.array(fit_data[::sample], dtype=np.int_)

    colnames = []
    params_label = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
                    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
                    'wd (deg)']

    # change the model
    params_label = params_label[:-1]
    # change the model

    for t in range(windows):
        for p in params_label:
            colnames.append((p, t))
    df = pd.DataFrame(fit_data[:800], columns=colnames)

    # A DBN with 3 stage
    dbn = DBN()

    edges = []

    for i in range(cc.shape[0]):
        for j in range(i + 1, cc.shape[1]):
            if (cc[i, j] >= 0.75) & (np.sum(cluster_l) <= 5):
                cluster_l[i] = 1
                cluster_l[j] = 1
                for t in range(windows):
                    edges.append(((params_label[i], t), (params_label[j], t)))

    for p in range(len(params_label)):
        if cluster_l[p] == 1:
            for t in range(windows - 1):
                edges.append(((params_label[p], t), (params_label[p], t + 1)))

    dbn.add_edges_from(edges)

    # p_14 = []
    # labels_14 = labels[:, -1]
    # for i in range(cluster_v[-1]):
    #     p_14.append([len(labels_14[labels_14 == i])/len(labels_14)])
    #
    # print("p_14: ")
    # print(p_14)
    # _14_cpd = TabularCPD(('wd (deg)', 0), cluster_v[-1], p_14)
    # dbn.add_cpds(_14_cpd)

    print("\nstart Train")
    batch_size = 2000

    for i in range(int(len(fit_data) / batch_size)):
        print("The " + str(i) + "th Times" + "\n")
        df = pd.DataFrame(fit_data[batch_size * i:batch_size * (i + 1)], columns=colnames)
        dbn.fit(df)

    pickle.dump(dbn, open(r"./model/dbn2_" + str(windows) + "_" + str(sample) + "_" + str(batch_size) + r".dat", 'wb'))
    pickle.dump(mycls, open(r"./model/cls2_" + str(windows) + "_" + str(sample) + "_" + str(batch_size) + r".dat", 'wb'))

    print("\nCheck Model:")
    print(dbn.check_model())
    print("\nCheck CPDs:")
    print(dbn.get_cpds())
    print("\nCheck Nodes:")
    print(dbn.nodes)
    print("\nCheck Edges:")
    print(dbn.edges)
    print("\nCheck Params:")
    print(cluster_l)
    print("\n\n")

    # [<TabularCPD representing P((rh (%), 0):6) at 0x7f18a2c4e640>,
    # <TabularCPD representing P((wv (m/s), 0):4 | (Tpot (K), 0):6, (VPdef (mbar), 0):4) at 0x7f18a307aca0>,
    # <TabularCPD representing P((VPmax (mbar), 0):6 | (T (degC), 0):6, (Tdew (degC), 0):10, (Tpot (K), 0):6) at 0x7f18a30a1ca0>,
    # <TabularCPD representing P((sh (g/kg), 0):8 | (T (degC), 0):6, (Tdew (degC), 0):10, (Tpot (K), 0):6, (VPact (mbar), 0):8, (VPdef (mbar), 0):4, (VPmax (mbar), 0):6) at 0x7f18a30ab9a0>,
    # <TabularCPD representing P((VPact (mbar), 0):8 | (T (degC), 0):6, (Tdew (degC), 0):10, (Tpot (K), 0):6, (VPmax (mbar), 0):6) at 0x7f18a3052dc0>,
    # <TabularCPD representing P((Tdew (degC), 0):10 | (T (degC), 0):6, (Tpot (K), 0):6) at 0x7f18a00d7ee0>,
    # <TabularCPD representing P(('wd (deg)', 0):8) at 0x7f18a3119d90>,
    # <TabularCPD representing P((VPdef (mbar), 0):4 | (T (degC), 0):6, (Tdew (degC), 0):10, (Tpot (K), 0):6, (VPact (mbar), 0):8, (VPmax (mbar), 0):6) at 0x7f18a3081fa0>,
    # <TabularCPD representing P((H2OC (mmol/mol), 0):8 | (T (degC), 0):6, (Tdew (degC), 0):10, (Tpot (K), 0):6, (VPact (mbar), 0):8, (VPdef (mbar), 0):4, (VPmax (mbar), 0):6, (sh (g/kg), 0):8) at 0x7f188f682220>,
    # <TabularCPD representing P((Tpot (K), 0):6 | (T (degC), 0):6) at 0x7f18a00d78e0>,
    # <TabularCPD representing P((max. wv (m/s), 0):4 | (T (degC), 0):6, (Tpot (K), 0):6, (VPdef (mbar), 0):4, (VPmax (mbar), 0):6, (wv (m/s), 0):4) at 0x7f18a00d70a0>,
    # <TabularCPD representing P((p (mbar), 0):6) at 0x7f188f6822e0>,
    # <TabularCPD representing P((T (degC), 0):6) at 0x7f18a00d7610>,
    # <TabularCPD representing P((rho (g/m**3), 0):6 | (p (mbar), 0):6, (rh (%), 0):6) at 0x7f18a00d00a0>,
    # <TabularCPD representing P((rh (%), 1):6 | (rh (%), 0):6) at 0x7f18a30ddc40>,
    # <TabularCPD representing P((wv (m/s), 1):4 | (Tpot (K), 1):6, (VPdef (mbar), 1):4, (wv (m/s), 0):4) at 0x7f18a307a370>,
    # <TabularCPD representing P((VPmax (mbar), 1):6 | (T (degC), 1):6, (Tdew (degC), 1):10, (Tpot (K), 1):6, (VPmax (mbar), 0):6) at 0x7f18a3052c10>,
    # <TabularCPD representing P((sh (g/kg), 1):8 | (T (degC), 1):6, (Tdew (degC), 1):10, (Tpot (K), 1):6, (VPact (mbar), 1):8, (VPdef (mbar), 1):4, (VPmax (mbar), 1):6, (sh (g/kg), 0):8) at 0x7f18a3052220>,
    # <TabularCPD representing P((VPact (mbar), 1):8 | (T (degC), 1):6, (Tdew (degC), 1):10, (Tpot (K), 1):6, (VPact (mbar), 0):8, (VPmax (mbar), 1):6) at 0x7f18a3081f40>,
    # <TabularCPD representing P((Tdew (degC), 1):10 | (T (degC), 1):6, (Tdew (degC), 0):10, (Tpot (K), 1):6) at 0x7f18a00d7bb0>,
    # <TabularCPD representing P((wd (deg), 1):8 | (wd (deg), 0):8) at 0x7f188fbe2790>,
    # <TabularCPD representing P((VPdef (mbar), 1):4 | (T (degC), 1):6, (Tdew (degC), 1):10, (Tpot (K), 1):6, (VPact (mbar), 1):8, (VPdef (mbar), 0):4, (VPmax (mbar), 1):6) at 0x7f18a3081bb0>,
    # <TabularCPD representing P((H2OC (mmol/mol), 1):8 | (H2OC (mmol/mol), 0):8, (T (degC), 1):6, (Tdew (degC), 1):10, (Tpot (K), 1):6, (VPact (mbar), 1):8, (VPdef (mbar), 1):4, (VPmax (mbar), 1):6, (sh (g/kg), 1):8) at 0x7f18a30a1160>,
    # <TabularCPD representing P((Tpot (K), 1):6 | (T (degC), 1):6, (Tpot (K), 0):6) at 0x7f18a00d0190>,
    # <TabularCPD representing P((max. wv (m/s), 1):4 | (T (degC), 1):6, (Tpot (K), 1):6, (VPdef (mbar), 1):4, (VPmax (mbar), 1):6, (max. wv (m/s), 0):4, (wv (m/s), 1):4) at 0x7f18a307a7f0>,
    # <TabularCPD representing P((p (mbar), 1):6 | (p (mbar), 0):6) at 0x7f18a30ab670>,
    # <TabularCPD representing P((T (degC), 1):6 | (T (degC), 0):6) at 0x7f18a00d7850>,
    # <TabularCPD representing P((rho (g/m**3), 1):6 | (p (mbar), 1):6, (rh (%), 1):6, (rho (g/m**3), 0):6) at 0x7f18a30c87f0>]

    dbn_infer = DBNInference(dbn)
