# TODO: 加载数据集

import numpy as np
import pandas as pd


def read_data(path, country):
    #! 加载指定国家的疫情数据
    data = pd.read_csv(path, encoding='utf-8')
    selected = data[data['Country'] == country]
    selected_dict = {
                    'Date': np.array(selected["Date_reported"]),
                    'Country': country,
                    'New_cases': np.array(selected["New_cases"]),
                    'New_deaths': np.array(selected["New_deaths"])
                    }
    return selected_dict



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    
    path = "data/WHO-COVID-19-global-data.csv"
    country = "China"

    data_dict = read_data(path, country)
    
    plt.figure()
    ax=plt.gca()
    plt.plot(np.log(data_dict["New_cases"]))
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # plt.tick_params(labelsize=5)
    # plt.xticks(rotation=90)
    plt.show()
