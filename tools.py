import numpy as np
import pandas as pd
import time
import os
from IDF_tool import Schedules, LocationClimate, MainFunctions, ThermalZonesSurfaces

def make_directory(nombre_caso, ruta_resultados):
    """
    Se crea un directorio para almacenar los resultados
    """
    fecha = str(time.strftime('%y-%m-%d'))
    hora = str(time.strftime('%H-%M'))
    caso = nombre_caso + str(time.strftime('%S'))
    directorio = ruta_resultados + '/' + fecha + '-'+ hora + '_'+ caso
    try:
        os.mkdir(directorio)
        os.mkdir(directorio+'/Resultados')
    except FileExistsError:
        time.sleep(3)
        caso = nombre_caso + str(time.strftime('%S'))
        directorio = ruta_resultados + '/' + fecha + '-'+ hora + '_'+ caso
        os.mkdir(directorio)
        os.mkdir(directorio+'/Resultados')
        print("Se ha creado el directorio: %s " % directorio)
    except:
        print("La creación del directorio %s falló" % directorio)
    else:
        print("Se ha creado el directorio: %s " % directorio)
    
    return directorio

def random_run_date_training():
        """
        Esta función devuelve un mes y un día aleatorios dentro del rango de los primeros
        20 días del mes.
        """
        month = int(np.random.randint(1, 13, 1))
        if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
            day = int(np.random.randint(1, 21, 1))
        elif month == 2:
            day = int(np.random.randint(1, 21, 1))
        else:
            day = int(np.random.randint(1, 21, 1))

        return month, day
    
def random_run_date_test():
        """
        Esta función devuelve un mes y un día aleatorios dentro del rango de los últimos
        días del mes, a partir del día 21.
        """
        month = int(np.random.randint(1, 13, 1))
        if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
            day = int(np.random.randint(21, 32, 1))
        elif month == 2:
            day = int(np.random.randint(21, 29, 1))
        else:
            day = int(np.random.randint(21, 31, 1))

        return month, day
    
def random_run_date():
        month = int(np.random.randint(1, 13, 1))
        if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
            day = int(np.random.randint(1, 32, 1))
        elif month == 2:
            day = int(np.random.randint(1, 29, 1))
        else:
            day = int(np.random.randint(1, 31, 1))

        return month, day

def plus_day(day, month, day_p):
    if month == 1 or 3 or 5 or 7 or 8 or 10 or 12:
            day_max = 31
    elif month == 2:
        day_max = 28
    else:
        day_max = 30
        
    if day_p != 0:
        
        if day + day_p > day_max:
            day += day_p - day_max
            if month != 12:
                month += 1
            else:
                month = 1
        else:
            day += day_p
    return day, month
    
def final_month_day(day, month):
    if month == 1 or 3 or 5 or 7 or 8 or 10 or 12:
            day_max = 31
    elif month == 2:
        day_max = 28
    else:
        day_max = 30
        
    day_p = 10
        
    if day + day_p > day_max:
        day += day_p - day_max
        if month != 12:
            month += 1
        else:
            month = 1
    else:
        day += day_p
        
    return day, month

def episode_epJSON(directorio, init_month, init_day, final_month=0, final_day=0):
        """
        Este toma un archivo epJSON y lo modifica. Luego retorna la ruta del archivo modificado.
        Las modificaciones efectuadas son:
            1. Cambio del mes a ejecutar.
            2. Cambio del día a ejecutar.
            3. Cambio en los path de los calendarios de disponibilidad de los objetos accionables en la vivienda.
        """
        if final_month == 0:
            final_month = init_month
        if final_day == 0:
            final_day = init_day
        epJSON_file_old = MainFunctions.MainFunctions.read_epjson(directorio + '/Resultados/modelo_simple.epJSON')
        LocationClimate.RunPeriod.begin_day_of_month(epJSON_file_old, "DDMM", init_day)
        LocationClimate.RunPeriod.begin_month(epJSON_file_old, "DDMM", init_month)
        LocationClimate.RunPeriod.end_day_of_month(epJSON_file_old, "DDMM", final_day)
        LocationClimate.RunPeriod.end_month(epJSON_file_old, "DDMM", final_month)
        Schedules.Schedule_File.file_name(epJSON_file_old, "Control_R", directorio + '/Resultados/Control_R_6.csv')
        Schedules.Schedule_File.file_name(epJSON_file_old, "Control_C", directorio + '/Resultados/Control_C_6.csv')
        Schedules.Schedule_File.file_name(epJSON_file_old, "Shadow_Control", directorio + '/Resultados/RL_Control_Sch_0_6.csv')
        Schedules.Schedule_File.file_name(epJSON_file_old, "VentN_Control", directorio + '/Resultados/VentN_Aviability_Sch_0_6.csv')
        Schedules.Schedule_File.file_name(epJSON_file_old, "VentS_Control", directorio + '/Resultados/VentS_Aviability_Sch_0_6.csv')
        MainFunctions.MainFunctions.write_epjson(directorio + '/Resultados/new.epJSON', epJSON_file_old)
        epJSON_new = directorio + '/Resultados/new.epJSON'
        return epJSON_new
    
def window_size_epJSON(epjson_path:str, window_name:str, factor:float):
        """
        Este toma un archivo epJSON y lo modifica. Luego retorna la ruta del archivo modificado.
        Las modificaciones efectuadas son:
            1. Cambio en la escala de las ventanas norte y sur.
        """
        epJSON_file_old = MainFunctions.MainFunctions.read_epjson(epjson_path)
        vertex_1_x_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_1_x_coordinate']
        vertex_1_y_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_1_y_coordinate']
        vertex_1_z_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_1_z_coordinate']
        vertex_2_x_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_2_x_coordinate']
        vertex_2_y_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_2_y_coordinate']
        vertex_2_z_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_2_z_coordinate']
        vertex_3_x_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_3_x_coordinate']
        vertex_3_y_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_3_y_coordinate']
        vertex_3_z_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_3_z_coordinate']
        vertex_4_x_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_4_x_coordinate']
        vertex_4_y_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_4_y_coordinate']
        vertex_4_z_coordinate_old = epJSON_file_old['FenestrationSurface:Detailed'][window_name]['vertex_4_z_coordinate']
        
        # Verificación de la simetría de la ventana
        if vertex_1_x_coordinate_old != vertex_2_x_coordinate_old:
            return "Error. La ventana no es cuadrada."
        if vertex_3_x_coordinate_old != vertex_4_x_coordinate_old:
            return "Error. La ventana no es cuadrada."
        if vertex_1_z_coordinate_old != vertex_4_z_coordinate_old:
            return "Error. La ventana no es cuadrada."
        if vertex_2_z_coordinate_old != vertex_3_z_coordinate_old:
            return "Error. La ventana no es cuadrada."
        
        # Verificación del plano de la ventana
        if vertex_1_y_coordinate_old != vertex_2_y_coordinate_old != vertex_3_y_coordinate_old != vertex_4_y_coordinate_old:
            return "Error. La ventana no se encuentra en un plano."
        
        area_old = (vertex_1_x_coordinate_old - vertex_4_x_coordinate_old)*(vertex_1_z_coordinate_old - vertex_2_z_coordinate_old)
        area_new = factor*area_old
        factor_edge = factor**(1/2)
        vertex_1_x_coordinate_new = vertex_1_x_coordinate_old * factor_edge
        vertex_4_x_coordinate_new = vertex_4_x_coordinate_old * factor_edge
        vertex_1_z_coordinate_new = vertex_1_z_coordinate_old * factor_edge
        vertex_2_z_coordinate_new = vertex_2_z_coordinate_old * factor_edge
        
        vertex_2_x_coordinate_new = vertex_1_x_coordinate_new
        vertex_3_x_coordinate_new = vertex_4_x_coordinate_new
        vertex_4_z_coordinate_new = vertex_1_z_coordinate_new
        vertex_3_z_coordinate_new = vertex_2_z_coordinate_new
        
        area_new_cheq = (vertex_1_x_coordinate_new - vertex_4_x_coordinate_new)*(vertex_1_z_coordinate_new - vertex_2_z_coordinate_new)
        
        error_scale = (area_new - area_new_cheq)/area_new * 100
        print(str(error_scale) + '%')
        
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_1_x_coordinate(
            epJSON_file_old, window_name, vertex_1_x_coordinate_new)
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_1_z_coordinate(
            epJSON_file_old, window_name, vertex_1_z_coordinate_new)
        
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_2_x_coordinate(
            epJSON_file_old, window_name, vertex_2_x_coordinate_new)
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_2_z_coordinate(
            epJSON_file_old, window_name, vertex_2_z_coordinate_new)
        
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_3_x_coordinate(
            epJSON_file_old, window_name, vertex_3_x_coordinate_new)
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_3_z_coordinate(
            epJSON_file_old, window_name, vertex_3_z_coordinate_new)
        
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_4_x_coordinate(
            epJSON_file_old, window_name, vertex_4_x_coordinate_new)
        ThermalZonesSurfaces.FenestrationSurface_Detailed.vertex_4_z_coordinate(
            epJSON_file_old, window_name, vertex_4_z_coordinate_new)
        
        MainFunctions.MainFunctions.write_epjson(epjson_path, epJSON_file_old)
        epJSON_new = epjson_path
        return epJSON_new
    
class weather_function():
    
    def climatic_stads(epw_file_path, day, month):
        epw_file = pd.read_csv(epw_file_path,
                            header = None, #['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Data Source and Uncertainty Flags', 'Dry Bulb Temperature', 'Dew Point Temperature', 'Relative Humidity', 'Atmospheric Station Pressure', 'Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation', 'Horizontal Infrared Radiation Intensity', 'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance', 'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Zenith Luminance', 'Wind Direction', 'Wind Speed', 'Total Sky Cover', 'Opaque Sky Cover', 'Visibility', 'Ceiling Height', 'Present Weather Observation', 'Present Weather Codes', 'Precipitable Water', 'Aerosol Optical Depth', 'Snow Depth', 'Days Since Last Snowfall', 'Albedo', 'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'],
                            skiprows = 8
                            )
        day_p1, month_p1 = plus_day(day, month, 1)
        day_p2, month_p2 = plus_day(day, month, 2)
        output = {
            'T_max_0': weather_function.tmax(epw_file, day, month),
            'T_min_0': weather_function.tmin(epw_file, day, month),
            'RH_0': weather_function.rh_avg(epw_file, day, month),
            'raining_total_0': weather_function.rain_tot(epw_file, day, month),
            'wind_avg_0': weather_function.wind_avg(epw_file, day, month),
            'wind_max_0': weather_function.wind_max(epw_file, day, month),
            'total_sky_cover_0': weather_function.total_sky_cover(epw_file, day, month),
            #average parameters of the day 1 after
            'T_max_1': weather_function.tmax(epw_file, day_p1, month_p1),
            'T_min_1': weather_function.tmin(epw_file, day_p1, month_p1),
            'RH_1': weather_function.rh_avg(epw_file, day_p1, month_p1),
            'raining_total_1': weather_function.rain_tot(epw_file, day_p1, month_p1),
            'wind_avg_1': weather_function.wind_avg(epw_file, day_p1, month_p1),
            'wind_max_1': weather_function.wind_max(epw_file, day_p1, month_p1),
            'total_sky_cover_1': weather_function.total_sky_cover(epw_file, day_p1, month_p1),
            #average parameters of the day 2 after,
            
            'T_max_2': weather_function.tmax(epw_file, day_p2, month_p2),
            'T_min_2': weather_function.tmin(epw_file, day_p2, month_p2),
            'RH_2': weather_function.rh_avg(epw_file, day_p2, month_p2),
            'raining_total_2': weather_function.rain_tot(epw_file, day_p2, month_p2),
            'wind_avg_2': weather_function.wind_avg(epw_file, day_p2, month_p2),
            'wind_max_2': weather_function.wind_max(epw_file, day_p2, month_p2),
            'total_sky_cover_2': weather_function.total_sky_cover(epw_file, day_p2, month_p2)
        }
        return output
        
    def tmax(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[6][_])
        tmax = max(array)
        return tmax


    def tmin(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[6][_])
        tmin = min(array)
        return tmin

    def rh_avg(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[8][_])
        rh_avg = sum(array)/len(array)
        return rh_avg

    def rain_tot(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[33][_])
        rain_tot = sum(array)
        return rain_tot

    def wind_avg(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[21][_])
        wind_avg = sum(array)/len(array)
        return wind_avg

    def wind_max(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[21][_])
        wind_max = max(array)
        return wind_max
    
    def total_sky_cover(epw_file, day, month, day_p=0):
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[22][_])
        total_sky_cover = sum(array)/len(array)
        return total_sky_cover
    
    
def HVAC_H_rew(q_H, To, Ti, PMV):
    """
    # HVAC Heating Reward
    This agent give penalties for use energy. When the temperature gap between
    the inferior limit of comfort range and outdoor is small the penalty is bigger.
    """
    e_H = (abs(q_H))/(3.6*1000000) # The energy consumption e is equal to the q_supp value but in kWh not in J
    
    if To > Ti and PMV < 0 and e_H > 0:
        HVAC_H_rew = -1
    elif To > Ti and PMV < 0 and e_H == 0:
        HVAC_H_rew = 0
    elif To < Ti and PMV < 0 and e_H > 0:
        HVAC_H_rew = 0
    elif To < Ti and PMV < 0 and e_H == 0:
        HVAC_H_rew = -1
    elif To > Ti and PMV > 0 and e_H > 0:
        HVAC_H_rew = -1
    elif To > Ti and PMV > 0 and e_H == 0:
        HVAC_H_rew = 0
    elif To < Ti and PMV > 0 and e_H > 0:
        HVAC_H_rew = -1
    elif To < Ti and PMV > 0 and e_H == 0:
        HVAC_H_rew = 0
    else:
        HVAC_H_rew = 0
    
    return HVAC_H_rew

def HVAC_C_rew(q_C, To, Ti, PMV):
    """
    # HVAC Cooling Reward
    This agent give penalties for use energy. When the temperature gap between
    the inferior limit of comfort range and outdoor is small the penalty is bigger.
    """
    e_C = (abs(q_C))/(3.6*1000000) # The energy consumption e is equal to the q_supp value but in kWh not in J

    if To > Ti and PMV < 0 and e_C > 0:
        HVAC_C_rew = -1
    elif To > Ti and PMV < 0 and e_C == 0:
        HVAC_C_rew = 0
    elif To < Ti and PMV < 0 and e_C > 0:
        HVAC_C_rew = -1
    elif To < Ti and PMV < 0 and e_C == 0:
        HVAC_C_rew = 0
    elif To > Ti and PMV > 0 and e_C > 0:
        HVAC_C_rew = 0
    elif To > Ti and PMV > 0 and e_C == 0:
        HVAC_C_rew = -1
    elif To < Ti and PMV > 0 and e_C > 0:
        HVAC_C_rew = -1
    elif To < Ti and PMV > 0 and e_C == 0:
        HVAC_C_rew = 0
    else:
        HVAC_C_rew = 0
    
    return HVAC_C_rew

def DSP_rew(q_H, q_C, To, Ti, PMV):
    DSP_rew = HVAC_H_rew(q_H, To, Ti, PMV) + HVAC_C_rew(q_C, To, Ti, PMV)
    return DSP_rew

def NW_rew(nw_state, To, Ti, PMV):
    """
    # North Window Reward
    The reward is based on the operation of the window.
    If the temperature outside is bigger than upper limit of comfort and inside is
    cool (Ti < T_dn) the window must be open but if inside is hot or neutral, window
    must be close.
    """
    if To > Ti and PMV < 0 and nw_state == 1:
        NW_rew = 0
    elif To > Ti and PMV < 0 and nw_state == 0:
        NW_rew = -1
    elif To < Ti and PMV < 0 and nw_state == 1:
        NW_rew = -1
    elif To < Ti and PMV < 0 and nw_state == 0:
        NW_rew = 0
    elif To > Ti and PMV > 0 and nw_state == 1:
        NW_rew = -1
    elif To > Ti and PMV > 0 and nw_state == 0:
        NW_rew = 0
    elif To < Ti and PMV > 0 and nw_state == 1:
        NW_rew = 0
    elif To < Ti and PMV > 0 and nw_state == 0:
        NW_rew = -1
    else:
        NW_rew = 0
    
    return NW_rew

def SW_rew(sw_state, To, Ti, PMV):
    """
    # South Window Reward
    The reward is based on the operation of the window.
    If the temperature outside is bigger than upper limit of comfort and inside is
    cool (Ti < T_dn) the window must be open but if inside is hot or neutral, window
    must be close.
    """
    if To > Ti and PMV < 0 and sw_state == 1:
        SW_rew = 0
    elif To > Ti and PMV < 0 and sw_state == 0:
        SW_rew = -1
    elif To < Ti and PMV < 0 and sw_state == 1:
        SW_rew = -1
    elif To < Ti and PMV < 0 and sw_state == 0:
        SW_rew = 0
    elif To > Ti and PMV > 0 and sw_state == 1:
        SW_rew = -1
    elif To > Ti and PMV > 0 and sw_state == 0:
        SW_rew = 0
    elif To < Ti and PMV > 0 and sw_state == 1:
        SW_rew = 0
    elif To < Ti and PMV > 0 and sw_state == 0:
        SW_rew = -1
    else:
        SW_rew = 0
    return SW_rew

def NWB_rew(nwb_state, PMV, rad):
    """
    # North Window Blind Reward
    # """
    if PMV > 0 and rad > 0 and nwb_state == 1:
        NWB_rew = 0
    elif PMV > 0 and rad > 0 and nwb_state == 0:
        NWB_rew = -1
    elif PMV < 0 and rad > 0 and nwb_state == 1:
        NWB_rew = -1
    elif PMV < 0 and rad > 0 and nwb_state == 0:
        NWB_rew = 0
    elif PMV > 0 and rad <= 0 and nwb_state == 1:
        NWB_rew = -1
    elif PMV > 0 and rad <= 0 and nwb_state == 0:
        NWB_rew = 0
    elif PMV < 0 and rad <= 0 and nwb_state == 1:
        NWB_rew = 0
    elif PMV < 0 and rad <= 0 and nwb_state == 0:
        NWB_rew = -1
    else:
        NWB_rew = 0
        
    return NWB_rew

def collaboratory_rew(beta, q_H, q_C, E_max, PMV):
    # Reward Multi-agent dictionary
    e_C = (abs(q_C))/(3.6*1000000) # The energy consumption e is equal to the q_supp value but in kWh not in J
    e_H = (abs(q_H))/(3.6*1000000) # The energy consumption e is equal to the q_supp value but in kWh not in J
    collaboratory_rew = -beta*(e_H + e_C)/(E_max) - (1 - beta)*((abs(PMV)**2)/9)
    return collaboratory_rew

def transform_centralized_action(central_action):
    centralized_action_space = np.loadtxt(
        'C:/Users/grhen/Documents/GitHub/EP_RLlib/centralized_action_space.csv',
        delimiter=',',
        skiprows=1,
        dtype=int
        )
    
    descentralized_action = [
        centralized_action_space[central_action][1],
        centralized_action_space[central_action][2],
        centralized_action_space[central_action][3],
        centralized_action_space[central_action][4],
        centralized_action_space[central_action][5]
    ]
    
    return descentralized_action