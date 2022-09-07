import pandas as pd

# with open('data/dewiki-20220201-clean.txt', 'r') as f:
#     while True:
#         next_line = f.readline()
#         if not next_line:
#             break
#         print(next_line.strip())
        

for i in range(6):
    df = pd.read_fwf('data/dewiki-20220201-clean.txt',
                    names=["sentences"],
                    widths=[-1],
                    skiprows=10000000*i,
                    nrows=10000000)
    print(df.shape)
    df.to_csv(f"data/dewiki-20220201-clean-0{i+1}.csv", index=False)