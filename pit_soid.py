import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import math
from tqdm import tqdm
import pandas as pd

# 1. 基础参数设置
V = 60000  # 储热水池体积 (m^3)
Cp_w = 4187  # 水的比热容 (J/(kg·K))
rho_w = 1000  # 水的密度 (kg/m^3)
lambda_w = 0.63  # 水的热导率 (W/(m·K))
lambda_soil = 1  # 土壤的热导率 (W/(m·K))
Cp_soil = 1800  # 土壤的比热容 (J/(kg·K))
rho_soil = 1800  # 土壤的密度 (kg/m^3)
lambda_ins = 0.44  # 绝缘层的热导率 (W/(m·K))
Cp_ins = 80  # 绝缘层的比热容 (J/(kg·K))
rho_ins = 80  # 绝缘层的密度 (kg/m^3)
U_top = 26.6  # 顶部热传递系数 (W/(m^2·K))
U_side = 111  # 侧壁热传递系数 (W/(m^2·K))
U_bot = 74.9  # 底部热传递系数 (W/(m^2·K))
T_amb = 35  # 环境温度 (°C)
T_soil = 25  # 土壤温度 (°C)
T_initial = 25  # 初始温度 (°C)
#T_inlet = 100  # 入口水温 (°C)
dt = 15  # 时间步长 (s)，优化后减少，防止数值不稳定
n_layers = 20  # 储热水池的层数
h = 16  # 水池高度 (m)
n_years = 1  # 模拟的年数
h_san = 45 * h / 32  # 虚拟三棱锥的高度
pi = 3.1415
m_flow = 20  # 水的质量流量 (kg/s)
alpha = 0.8 #常见地面吸收系数Absorption factor of ground surface
G = 800 #Global irradiance(w/m^2)

# 2. 网格划分
dz = 0.8  # 每层高度 (m)
A_top = 8100  # 顶部面积 (m^2)
A_bot = 676  # 底部面积 (m^2)

# 初始化温度场 (强制使用 float64)
T_water = np.full(n_layers + 2, T_initial, dtype=np.float64)
T_soil_layers = np.full(n_layers + 1, T_soil, dtype=np.float64)
D_lay = np.ones(n_layers + 1, dtype=np.float64)
A_b = np.ones(n_layers + 1, dtype=np.float64)
A_side = np.ones(n_layers + 1, dtype=np.float64)
h_ever_san = np.ones(n_layers + 1, dtype=np.float64)
K_lay = np.ones(n_layers + 1, dtype=np.float64)
V_lay = np.ones(n_layers + 1, dtype=np.float64)

#初始化土壤
n_r = 100  # r 方向网格数
n_z = 40  # z 方向网格数
dr = 1.0  # r 方向步长
dz_soil = 0.8  # z 方向步长
T_soil_grid = np.full((n_r + 2, n_z + 2), T_soil, dtype=np.float64)
K_z = np.full((n_r + 2, n_z + 2), 0, dtype=np.float64)
K_r = np.full((n_r + 2, n_z + 2), 0, dtype=np.float64)
C_s = np.full((n_r + 2, n_z + 2), Cp_soil, dtype=np.float64)

# 3. 边界初始化
D_lay[0] = 90
D_lay[n_layers] = 26
A_b[n_layers] = D_lay[n_layers] ** 2
h_ever_san[n_layers] = h_san - h

# 几何参数（与你图示一致：上口90，下口26，总高16 m）
H = 16.0
W_top = 90.0   # 上口边长
W_bot = 26.0   # 下口边长

def side_len(z):  # z从0(顶)到H(底)沿深度
    return W_top - (W_top - W_bot) * (z / H)

for j in range(1, n_layers + 1):
    z0 = (j - 1) * dz
    z1 = j * dz
    s0 = side_len(z0)
    s1 = side_len(z1)
    A0 = s0 * s0  # 上截面积
    A1 = s1 * s1  # 下截面积
    # 截棱锥体积（严谨公式）
    V_layer = dz / 3.0 * (A0 + A1 + math.sqrt(A0 * A1))
    V_lay[j] = V_layer
    A_b[j] = A1  # 若需要每层“底面”面积的近似（可选）
    A_side[j] = pi * math.sqrt(dz ** 2 + (s0 - s1) ** 2 / 4) * (s0 + s1) / 2  # 侧壁面积
    K_lay[j] = pi * s1 * s1 * lambda_w / dz / 4

K_top = lambda_ins * A_top / dz  # 顶部热传导系数
K_bot = A_b[-1] / (dz / 2 / lambda_soil + 1 / U_bot)  # 底部热传导系数

T_water[-1] = T_soil
K_lay[0] = K_top
K_lay[-1] = K_bot

for i in range(1, n_z + 1):
    for k in range(1, n_r + 1):
        for x in range(n_r + 1):  # 初始化边界温度
            T_soil_grid[x][1] = T_water[1]
            T_soil_grid[x][0] = (alpha * G + U_top * T_amb + 2 * T_soil_grid[x][1] * lambda_soil / dz_soil) / (
                        U_top + 2 * lambda_soil / dz_soil)
            T_soil_grid[k][-1] = 8.3
        for y in range(n_z + 1):
            T_soil_grid[-1][y] = 8.3
        r_k = k * dr
        r_k1 = (k + 1) * dr
        A_k_i = pi * ((r_k + dr / 2) ** 2 - (r_k - dr / 2) ** 2)
        # 判定是绝缘层还是土壤
        if i == 0 and 0 < k < 45:  # 绝缘层
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_ins
            K_z[k][i] = lambda_ins / A_k_i
            K_r[k][i] = 2 * pi * dz_soil * lambda_ins / math.log(r_k1 / r_k)
        elif i == 0 and k == 0:
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_ins
            K_z[k][i] = 0
            K_r[k][i] = 2 * pi * dz_soil * lambda_ins / math.log(r_k1 / r_k)
        elif i == 0 and k == n_r:
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_ins
            K_z[k][i] = 0
            K_r[k][i] = 2 * pi * dz_soil * lambda_ins / math.log(r_k1 / r_k)
        elif i != 0 and k == 0:
            C_s[k][i] = A_k_i * dz_soil * Cp_soil * rho_soil
            K_z[k][i] = 0
            K_r[k][i] = 2 * pi * dz_soil * lambda_soil / math.log(r_k1 / r_k)
        elif i != 0 and k == n_r:
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_ins
            K_z[k][i] = 0
            K_r[k][i] = 2 * pi * dz_soil * lambda_ins / math.log(r_k1 / r_k)
        elif i == 1:
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_soil
            K_z[k][i] = lambda_soil / A_k_i
            K_r[k][i] = 0
        elif i == n_z:
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_soil
            K_z[k][i] = lambda_soil / A_k_i
            K_r[k][i] = 0
        elif i == 0 and k == 45:  # 绝缘层与土壤的边界
            C_s[k][i] = A_k_i * dz_soil * Cp_ins * rho_ins
            K_z[k][i] = lambda_ins / A_k_i
            K_r[k][i] = 2 * pi * dz_soil / (
                        math.log((r_k + dr / 2) / r_k) / lambda_ins + math.log(r_k1 / ((r_k + dr / 2))) / lambda_soil)
        elif i == 16 and 0 < k <= 23:  # 水池底部与土壤接触层
            C_s[k][i] = A_k_i * dz_soil * Cp_soil * rho_soil
            K_z[k][i] = A_k_i / (0.5 * dz / lambda_soil + 1 / U_bot)
            K_r[k][i] = 2 * pi * dz_soil * lambda_soil / math.log(r_k1 / r_k)
        elif 1 < i <= 16 and i <= -0.5 * k + 23.5:
            continue
        else:
            C_s[k][i] = A_k_i * dz_soil * Cp_soil * rho_soil
            K_z[k][i] = lambda_soil / A_k_i
            K_r[k][i] = 2 * pi * dz_soil * lambda_soil / math.log(r_k1 / r_k)

day = 80

# 4. 计算模拟时间步数
time_steps = int(n_years * day * 24 * 3600 / dt)

# 存储平均温度
average_temperatures = []
aver_temp_highs = []
Q_st = []
df = pd.read_csv("temperature_at_35km.csv")

# 5. 模拟循环
for t in tqdm(range(time_steps), desc="模拟进度"):
    def get_temperature_at_35km(ttime):
        """ 获取对应时间的 T_35km 温度 """
        return df[df['time'] == int(ttime * 15)]['T_35km'].values[0]


    # dT = 3889 * (get_temperature_at_35km(t) - 45.5) / 120
    # T_inlet = min(dT + T_water[1], 105)  # 初始入口温度
    T_inlet = 90

    for j in range(1, n_layers + 1):

        # 水池内温度变化
        M_io = 0
        M_oi = m_flow * Cp_w
        #print(f"Step {t}, Layer {j}, K_lay[j]={K_lay[j]}, K_lay[j-1]={K_lay[j - 1]}, M_oi={M_oi}, M_io={M_io}")
        if np.isnan(T_water[j - 1]) or np.isnan(T_water[j + 1]) or np.isnan(T_water[j]):
            print(
                f"NaN 发生在: Step {t}, Layer {j}, T_water[j-1]={T_water[j - 1]}, T_water[j+1]={T_water[j + 1]}, T_water[j]={T_water[j]}")
            break

        f1 = -K_lay[j] - K_lay[j - 1] - U_side * A_side[j] - M_oi
        f2 = (K_lay[j]) * T_water[j + 1] + (K_lay[j - 1]) * T_water[j - 1] + M_oi * T_water[j - 1] + A_side[j] * U_side * T_soil_grid[math.ceil(45 - 2 * j)][j]

        if j == 1:  # 顶层
            f2 += m_flow * Cp_w * (T_inlet - T_water[j - 1])

        #print(f"Step {t}, Layer {j}, f2={f2}, T_inlet={T_inlet}, T_water[j]={T_water[j]}")

        # 计算温度变化，并限制变化范围，防止过大导致溢出
        delta_T = (f1 * T_water[j] + f2) * dt / (V_lay[j] * Cp_w * rho_w)
        #print(delta_T)
        #T_water[j] += np.clip(delta_T, -5, 5)  # 限制单步变化范围
        T_water[j] += delta_T


        #土壤温度变化
    for i in range(1, n_z + 1):
        for k in range(1, n_r + 1):
            if k <= 45 and i <= 16 and i > 0.5 * k + 23.5:
                f1_s = 2 * (-K_r[k][i] - K_z[k][i] - A_side[i] - A_side[i + 1] * U_side)
                f2_s = 2 * (K_z[k][i] * T_soil_grid[k + 1][i] + K_r[k][i] * T_soil_grid[k][i + 1] + A_side[i] * U_side * T_water[i] + A_side[i + 1] * U_side * T_water[i + 1])
            else:
                f1_s = -K_z[k][i] - K_z[k][i - 1] - K_r[k][i] - K_r[k - 1][i]
                f2_s = K_z[k][i] * T_soil_grid[k][i + 1] + K_z[k][i - 1] * T_soil_grid[k][i - 1] + K_r[k][i] * T_soil_grid[k + 1][i] + K_r[k - 1][i] * T_soil_grid[k - 1][i]

            delta_Ts = (f1_s * T_soil_grid[k][i] + f2_s) * dt / C_s[k][i]

            T_soil_grid[k][i] += delta_Ts
    dQ_st = 0
    # 计算水池热量变化
    for i in range(n_layers):
        dQ_st += (T_water[i] - T_initial) * Cp_w * V_lay[i] * rho_w

    Q_st.append(dQ_st)


    # 计算平均温度
    average_temperature = np.mean(T_water)
    average_temperatures.append(average_temperature)

    aver_temp_high = np.mean(T_water[1:4])
    aver_temp_highs.append(aver_temp_high)



# 6. 绘制温度变化曲线
time = np.linspace(0, n_years * day, len(average_temperatures))  # 时间 (天)
plt.plot(time, average_temperatures, label='Simulated Average Temperature')
plt.xlabel('Time (days)')
plt.ylabel('Average Temperature (°C)')

plt.legend()
plt.show()

time = np.linspace(0, n_years * day, len(aver_temp_highs))  # 时间 (天)
plt.plot(time, aver_temp_highs, label='Simulated Average Temperature')
plt.xlabel('Time (days)')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.show()

# 热量变化
energy = np.linspace(0, n_years * day, len(Q_st))  # 时间 (天)
plt.plot(energy, Q_st, label='Simulated Energy Change')
plt.xlabel('Time (days)')
plt.ylabel('Energy Change (J)')

plt.legend()
plt.show()

# 替换为你的模拟结果
final_temperatures = T_water[1:n_layers+1]

# 参数定义
H = 45  # 总高
W_top = 90  # 顶部宽
W_bot = 26  # 底部宽
n_layers = len(final_temperatures)
layer_heights = np.linspace(0, H, n_layers + 1)

# 每层顶部宽度按高度线性插值
widths = W_bot + (W_top - W_bot) * (layer_heights[:-1] / H)
widths_next = W_bot + (W_top - W_bot) * (layer_heights[1:] / H)

# 颜色映射（红热蓝冷）
norm = plt.Normalize(vmin=final_temperatures.min(), vmax=final_temperatures.max())
cmap = plt.get_cmap("coolwarm")

# 绘图
fig, ax = plt.subplots(figsize=(6, 12))
for i in range(n_layers):
    temp = final_temperatures[n_layers - 1 - i]
    color = cmap(norm(temp))
    poly = patches.Polygon([
        [-(widths[i] / 2), layer_heights[i]],
        [widths[i] / 2, layer_heights[i]],
        [widths_next[i] / 2, layer_heights[i + 1]],
        [-(widths_next[i] / 2), layer_heights[i + 1]]
    ], closed=True, facecolor=color, edgecolor='black')
    ax.add_patch(poly)

# 设置图像比例精确匹配真实比例
ax.set_xlim(-W_top / 2 - 5, W_top / 2 + 5)
ax.set_ylim(0, H)
ax.set_aspect('equal')  # 关键：保证图形真实等比例
ax.axis('off')  # 去除坐标轴

# 添加颜色条
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.01)
cbar.set_label('Temperature (°C)')

# 保存图像
plt.savefig("true_scale_temperature_trapezoid.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# 创建 DataFrame
df = pd.DataFrame({
    'Time (days)': time,
    'Average Temperature (°C)': average_temperatures
})
df0 = pd.DataFrame({
    'Time (days)': time,
    'Average Temperature (°C)': aver_temp_highs
})
# 创建 DataFrame
df2 = pd.DataFrame({
    'Time (days)': time,
    'Energy Change (J)': Q_st
})
# 保存到 Excel 文件

df.to_excel("simulation_results_soil.xlsx", index=False)
df0.to_excel("simulation_results_soil2.xlsx", index=False)
df2.to_excel("simulation_results_energy.xlsx", index=False)
