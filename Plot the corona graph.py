import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd

style.use('fivethirtyeight')
fig =plt.figure()
ax1 =fig.add_subplot(2,2,1)
ax2 =fig.add_subplot(2,2,2)
ax3 =fig.add_subplot(2,2,3)
ax4 =fig.add_subplot(2,2,4)

df = pd.read_csv('real time corona death in India.csv')

print(df)

# df['Unnamed: 2','Unnamed: 3'] = df['Unnamed: 2','Unnamed: 3'].astype(float)
#
# print(df['Unnamed: 2','Unnamed: 3'])
# df(['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5']) = df['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5'].astype(float)

def animate(i):
    df = pd.read_csv('real time corona death in India.csv')

    ys =df.iloc[1:,2].values
    xs = list(range(1,len(ys)+1))
    ax1.clear()
    ax1.plot(xs,ys)
    ax1.set_title('Corona 1',fontsize =12)

    ys = df.iloc[1:,3].values
    ax2.clear()
    ax2.plot(xs, ys)
    ax2.set_title('Corona 2', fontsize=12)

    ys = df.iloc[1:,4].values
    ax3.clear()
    ax3.plot(xs, ys)
    ax3.set_title('Corona 3', fontsize=12)

    ys = df.iloc[1:,5].values
    ax4.clear()
    ax4.plot(xs, ys)
    ax4.set_title('Corona 4', fontsize=12)

ani =animation.FuncAnimation(fig,animate,interval=1000)

plt.tight_layout()
plt.show()



