import pandas as pd
import numpy as np
# final_ssim_info = [['1'], [0.5164262], ['a']]
final_ssim_info = ['1', 0.5164262]
final_mse_info = ['1', 1420]
data = [['1', '51', 0.5164262, 1420]]
values = np.array(final_ssim_info)
values = np.resize(values, (1,2))
avalues = np.array(final_mse_info)
avalues = np.resize(avalues, (1,2))
print(values)
print("final_ssim_info:", final_ssim_info)
print("final_mse_info:", final_mse_info)
temp_column_header = ['Clean Image indexes vs Threat Image indexes']
temp_column__header_2 = [str(pair_of_data[1]) for pair_of_data in data]
joined_column_header = temp_column_header + temp_column__header_2
print("joined_column_header:", joined_column_header)
df_ssim = pd.DataFrame(values, columns=joined_column_header)
df_mse = pd.DataFrame(avalues, columns=[joined_column_header])
# columns=[str(i+1) for i in range(3)]
# fake_joined_column_header = ['Clean Image indexes vs Threat Image indexes', '51']
# print(fake_joined_column_header==joined_column_header)