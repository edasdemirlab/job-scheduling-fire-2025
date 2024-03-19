import random
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd


class UserInputsRead:
    def __init__(self):

        # read problem input
        self.directory = os.path.join('inputs', 'case_instance_inputs.csv')  # os.getcwd(),
        self.case_input_df = pd.read_csv(self.directory, index_col=0).dropna(axis=0, how='all').dropna(axis=1, how='all')


def generate_fire_degradation_rate(veg, fire_degradation_rates):
    # Define the range for fire_degradation_rate based on the veg parameter
    min_rate = fire_degradation_rates['min_rate']
    max_rate = fire_degradation_rates['max_rate']
    # Interpolate linearly between the min_rate and max_rate based on veg
    fire_degradation_rate = min_rate + veg * (max_rate - min_rate)

    return fire_degradation_rate

def normalize_value(original_value, min_value, max_value):
    return (original_value - min_value) / (max_value - min_value)

def get_rgb_val(number):
    if number == 0:
        return (0, 100, 150)
    elif number == -1:
        return (255, 255, 255)
    elif 1 <= number <= 10:
        # Calculate the green shade based on the input number
        norm_number = normalize_value(number, 1, 10)
        green_value = 200 - int(100 * norm_number)
        return (25, green_value, 75)
    else:
        raise ValueError("Number must be 0, -1, or in the range of 1 to 10")

def get_rgb_deg(number):
    norm_number = normalize_value(number, 0.8, 8)
    red_value = 200 - int(150 * norm_number)
    return (red_value, 25, 25)


def generate_grid(user_inputs):
    init_fire = int(user_inputs.parameters_df.loc["number_of_initial_fires", "value"])
    n = int(user_inputs.parameters_df.loc["number_of_grids_at_a_side", "value"])
    m = int(user_inputs.parameters_df.loc["number_of_areas_of_different_types", "value"])
    number_of_water_bodies = int(user_inputs.parameters_df.loc["number_of_water_bodies", "value"])
    number_of_blocks = int(user_inputs.parameters_df.loc["number_of_blocks", "value"])
    default_housing_density = int(user_inputs.parameters_df.loc["default_housing_density", "value"])
    default_vegetation_density = int(user_inputs.parameters_df.loc["default_vegetation_density", "value"])
    default_density = (default_housing_density, default_vegetation_density)
    # if include_water == 1:
    #     water = True
    # else:
    #     water = False
    # if include_block == 1:
    #     block = True
    # else:
    #     block = False

    csv_filename = user_inputs.case_output_file_name

    fire_degradation_rate_min = float(user_inputs.parameters_df.loc["fire_degradation_rate_min", "value"])
    fire_degradation_rate_max = float(user_inputs.parameters_df.loc["fire_degradation_rate_max", "value"])
    fire_degradation_rates = {'min_rate': fire_degradation_rate_min, 'max_rate': fire_degradation_rate_max}

    # set init_fire to be number of nodes you want to start the fire at initially.
    # n is dimension of square grid, m is number of areas of different types to generate
    # Create an empty grid
    grid = [[None] * n for _ in range(n)]

    # Define the area types
    area_types = [
        {'housing_density': 1, 'vegetation_density': 0},
        {'housing_density': 1, 'vegetation_density': 1},
        {'housing_density': 2, 'vegetation_density': 0},
        {'housing_density': 2, 'vegetation_density': 1},
        {'housing_density': 3, 'vegetation_density': 0},
        {'housing_density': 3, 'vegetation_density': 1},
        {'housing_density': 4, 'vegetation_density': 0},
        {'housing_density': 4, 'vegetation_density': 1},
        {'housing_density': 5, 'vegetation_density': 0},
        {'housing_density': 5, 'vegetation_density': 1}
    ]
    
    water_area_type = {'housing_density': 0, 'vegetation_density': 2}
    block_area_type = {'housing_density': -1, 'vegetation_density': 3}
    


    # Create m rectangular areas
    for area_number in range(m):
        if area_number >= m-number_of_water_bodies:
            area_type = water_area_type
        elif area_number < m-number_of_water_bodies and area_number >= (m - number_of_water_bodies - number_of_blocks):
            area_type = block_area_type
        else:
            area_type = random.choice(area_types)
        area_width = random.randint(1, n // 3)  # Random width (at most 1/3 of n)
        area_height = random.randint(1, n // 3)  # Random height (at most 1/3 of n)
        start_row = random.randint(0, n - area_height)  # Random starting row index
        start_col = random.randint(0, n - area_width)  # Random starting column index

        # Assign the area type to the grid cells
        for i in range(start_row, start_row + area_height):
            for j in range(start_col, start_col + area_width):
                grid[i][j] = area_type

    # Fill in the remaining cells with the default area type
    default_area = {'housing_density': default_density[0], 'vegetation_density': default_density[1]}
    for i in range(n):
        for j in range(n):
            if grid[i][j] is None:
                grid[i][j] = default_area
    
    grid[0][0] = {'housing_density': 1, 'vegetation_density': 0}

    # Generate CSV data
    csv_data = []
    all_coordinates = [(x, y) for x in range(n) for y in range(n)]
    random.shuffle(all_coordinates)
    for (i, j) in all_coordinates:
        area = grid[i][j]
        node_id = i*n + j + 1

        # coordinates are assigned as centers of cells
        x_coordinate = j + 0.5
        y_coordinate = i + 0.5

        # value at start is assigned to be higher as housing density increases
        housing_density = area['housing_density']
        if housing_density == 0:  # water
            value_at_start = 0
        elif housing_density == -1:  # block
            value_at_start = -1
        elif housing_density == 1:  # 0
            value_at_start = 1
        elif housing_density == 2:  # <6
            value_at_start = random.uniform(1, 2)
        elif housing_density == 3:  # 6-50
            value_at_start = random.uniform(2, 4)
        elif housing_density == 4:  # 50-741
            value_at_start = random.uniform(4, 10)
        elif housing_density == 5:  # >741
            value_at_start = 10

        # "veg" is randomly assigned to be b/w 0 and 0.5 if vegetation density is 0, and b/w 0.5 and 1 otherwise
        vegetation_density = area['vegetation_density']
        if vegetation_density == 2 or vegetation_density == 3:  # water or block
            veg = 0
        elif vegetation_density == 0:
            veg = round(random.uniform(0, 0.5), 2)
        else:
            veg = round(random.uniform(0.5, 1), 2)


        # then, fire degradation rate and fire amelioration rate are assigned based on veg; higher veg - faster fire
        if vegetation_density == 2 or vegetation_density == 3:  # water or block
            fire_degradation_rate = 0
            fire_amelioration_rate = 0
        else:
            fire_degradation_rate = generate_fire_degradation_rate(veg, fire_degradation_rates)  # value b/w 0.8 and 8
            fire_amelioration_rate = round(random.uniform(fire_degradation_rate - 0.5, fire_degradation_rate + 0.5),
                                           2) / 2  # value b/w 0.4 and 4
            if fire_amelioration_rate > fire_degradation_rate_max/2:
                fire_amelioration_rate = fire_degradation_rate_max/2
            elif fire_amelioration_rate <= fire_degradation_rate_min:
                fire_amelioration_rate = fire_degradation_rate_min/2
         # value degradation is also assigned based on the above rates: faster the fire, faster the value deteriorates
        if vegetation_density == 2 or vegetation_density == 3:  # water or block
            value_degradation_rate = 0
        else:
            value_degradation_rate = value_at_start / (1 / fire_degradation_rate + 1 / fire_amelioration_rate)


        neighborhood_list = []
        if i > 0:
            neighborhood_list.append((i - 1) * n + j + 1)  # Above cell
        if i < n - 1:
            neighborhood_list.append((i + 1) * n + j + 1)  # Below cell
        if j > 0:
            neighborhood_list.append(i * n + j)  # Left cell
        if j < n - 1:
            neighborhood_list.append(i * n + j + 2)  # Right cell

        # assigning state value:
        if x_coordinate == 0.5 and y_coordinate == 0.5:
            state = 6
        elif vegetation_density == 2:
            state = 5
        elif vegetation_density == 3:
            state = 4
        elif init_fire > 0:
            state = 1
            init_fire -= 1
        else:
            state = 0

        csv_data.append([
            node_id, x_coordinate, y_coordinate, value_at_start,
            value_degradation_rate, fire_degradation_rate,
            fire_amelioration_rate, neighborhood_list, state
        ])

    # Write CSV file
    if csv_filename:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'node_id', 'x_coordinate', 'y_coordinate', 'value_at_start',
                'value_degradation_rate', 'fire_degradation_rate',
                'fire_amelioration_rate', 'neighborhood_list', 'state'
            ])
            writer.writerows(csv_data)

    # Plot the grid
    plt.figure(figsize=(6, 6))
    plt.imshow([[area['housing_density'] for area in row] for row in grid], cmap='RdYlBu_r', vmin=0, vmax=5,
               origin='lower')
    plt.colorbar(label='Housing Density')
    veg = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            veg[i][j] = grid[i][j]['vegetation_density']
    for i in range(n):
        for j in range(n):
            if veg[i][j] == 2:
                plt.annotate('w', xy=(j, i), ha='center', va='center')
            elif veg[i][j] == 3:
                plt.annotate('b', xy=(j, i), ha='center', va='center')
            else:
                plt.annotate(veg[i][j], xy=(j, i), ha='center', va='center')

    plt.title('Density Grid')
    # plt.show()
    plt.savefig(csv_filename.split('.', 1)[0]+'_density_map.png')


    df_unsorted = pd.DataFrame(csv_data)
    df = df_unsorted.sort_values(by=df_unsorted.columns[0], ascending=True)


    # Generate RBG map based on value_at_start
    val = [[None] * n for _ in range(n)]
    idd = 0
    for i in range(n):
        for j in range(n):
            val[i][j] = get_rgb_val(df.iloc[idd][3])
            idd += 1

    # Generate RGB map based on fire_degradation_rate

    deg = [[None] * n for _ in range(n)]
    idd = 0
    for i in range(n):
        for j in range(n):
            if df.iloc[idd][5] == 0 and df.iloc[idd][3] == 0:
                deg[i][j] = (0, 100, 150)
            elif df.iloc[idd][5] == 0 and df.iloc[idd][3] == -1:
                deg[i][j] = (255, 255, 255)
            else:
                deg[i][j] = get_rgb_deg(df.iloc[idd][5])
            idd += 1

    # Plot value_at_start
    plt.figure(figsize=(6, 6))
    plt.imshow(val, origin="lower")
    idd = 0
    for i in range(n):
        for j in range(n):
            plt.annotate(df.iloc[idd][0], xy=(j, i), ha='center', va='center')
            idd += 1
    plt.title('value_at_start')
    # plt.show()
    plt.savefig(csv_filename.split('.', 1)[0]+'_value_map.png')

    # Plot fire_degradation_rate
    plt.figure(figsize=(6, 6))
    plt.imshow(deg, origin="lower")
    idd = 0
    for i in range(n):
        for j in range(n):
            plt.annotate(df.iloc[idd][0], xy=(j, i), ha='center', va='center', color = "white")
            if df.iloc[idd][8] == 1:
                plt.annotate("X", xy=(j + 0.4, i - 0.4), ha='right', va='bottom', color="yellow")
            idd += 1
    plt.title('fire_degradation_rate')
    # plt.show()
    plt.savefig(csv_filename.split('.', 1)[0]+'_fire_spread_map.png')

    plt.close('all')


