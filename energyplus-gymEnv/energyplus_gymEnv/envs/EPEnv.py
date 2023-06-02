"""
EnergyPlus Gym Environment configuration
"""
import sys
import os
sys.path.insert(0, 'C:/EnergyPlusV23-1-0')
import threading

from typing import Dict, Any, Optional, List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange
from queue import Queue, Empty, Full
from pythermalcomfort.models import pmv

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


class EnergyPlusRunner:

    def __init__(self, episode: int, env_config: Dict[str, Any], obs_queue: Queue, act_queue: Queue) -> None:
        self.episode = episode
        self.env_config = env_config
        self.obs_queue = obs_queue
        self.act_queue = act_queue

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.init_queue = Queue()
        self.progress_value: int = 0
        self.simulation_complete = False
        
        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = {
            # radiation in the plane of the windows
            "Bw": ("Surface Outside Face Incident Solar Radiation Rate per Area", "Zn001:Wall001:Win001"),
            # outside temperature
            "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
            # inside (zone) temperature
            "Ti": ("Zone Mean Air Temperature", "Thermal Zone: Modelo_Simple"),
            # wind speed in the site
            "v": ("Site Wind Speed", "Environment"),
            # direction of the wind
            "d": ("Site Wind Direction", "Environment"),
            # inside relativ humidity
            "RHi": ("Zone Air Relative Humidity", "Thermal Zone: Modelo_Simple")
        }
        self.var_handles: Dict[str, int] = {}

        self.meters = {
            # HVAC elec (J)
            "dh": "Heating:DistrictHeating",
            # District heating (J)
            "dc": "Cooling:DistrictCooling"
        }
        self.meter_handles: Dict[str, int] = {}

        self.actuators = {
            # handle para el control de la persiana
            "Shading": ('Schedule Value', 'Shadow_Control'),
            # control del refrigerador
            "Cooling": ('Schedule Value', 'Control_R'),
            # control del calefactor
            "Heating": ('Schedule Value', 'Control_C'),
            # control de abertura de la ventana orientada al norte
            "VentN": ('Schedule Value', 'VentN_Control'),
            # control de abertura de la ventana orientada al sur
            "VentS": ('Schedule Value', 'VentS_Control')
        }
        self.actuator_handles: Dict[str, int] = {}

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime

        # register callback used to track simulation progress
        def report_progress(progress: int) -> None:
            self.progress_value = progress

        runtime.callback_progress(self.energyplus_state, report_progress)
        """
        This function allows a client to register a function to be called back by EnergyPlus at 
        the end of each day with a progress (percentage) indicator
        """        
        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            print(f"running EnergyPlus with args: {cmd_args}")

            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)

        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.energyplus_api.runtime,
                self.make_eplus_args(),
                self.energyplus_state,
                self.sim_results
            )
        )
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if self.energyplus_exec_thread:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """
        make command line arguments to pass to EnergyPlus
        """
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += [
            "-w",
            self.env_config["epw"],
            "-d",
            f"{self.env_config['output']}/episode-{self.episode:08}-{os.getpid():05}",
            self.env_config["idf"]
        ]
        return eplus_args

    def _collect_obs(self, state_argument) -> None:
        """
        EnergyPlus callback that collects output variables/meters
        values and enqueue them
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_obs = {
            **{
                key: self.x.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handles.items()
            },
            **{
                key: self.x.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handles.items()
            }
        }
        time_step = self.x.zone_time_step_number(state_argument)
        hour = self.x.hour(state_argument)
        if hour < 23:
            To_p1h = self.x.today_weather_outdoor_dry_bulb_at_time(state_argument, hour+1, time_step)
        else:
            To_p1h = self.x.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour-23, time_step)
        if hour < 22:
            To_p2h = self.x.today_weather_outdoor_dry_bulb_at_time(state_argument, hour+2, time_step)
        else:
            To_p2h = self.x.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour-22, time_step)
        if hour < 21:
            To_p3h = self.x.today_weather_outdoor_dry_bulb_at_time(state_argument, hour+3, time_step)
        else:
            To_p3h = self.x.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour-21, time_step)
        day = self.x.day_of_month(state_argument)
        month = self.x.month(state_argument)
        climatic_stads = weather_function.climatic_stads(self.env_config["epw"], day, month)
        
        '''Comfort Range'''
        # Lectura de la temperatura media radiante del ambiente
        Ti_rad_handle = self.x.get_variable_handle(state_argument, "Zone Mean Radiant Temperature", "Thermal Zone: Modelo_Simple")
        Ti_rad = self.x.get_variable_value(state_argument, Ti_rad_handle)
        # Cálculo del met
        if hour >= 23 or hour < 7:
            met = 1
        else:
            met = 2
        # Cálculo del clo adaptativo
        To_at_6am = self.x.today_weather_outdoor_dry_bulb_at_time(state_argument, 6, 0)
        if To_at_6am <= -5:
            clo = 1.0
        elif To_at_6am <= 5:
            clo = 0.8 - To_at_6am*(0.15/5)
        elif To_at_6am <= 26:
            clo = 0.7 + To_at_6am*(0.25/26)
        else:
            clo = 0.45
        
        PMV = pmv(self.next_obs["Ti"], Ti_rad, 0.1, self.next_obs["RHi"], met, clo, 0, "ASHRAE")
        
        
        self.next_obs.update(
            {
            "rad": self.x.today_weather_beam_solar_at_time(state_argument, hour, time_step),
            "To_p1h": To_p1h,
            "To_p2h": To_p2h,
            "To_p3h": To_p3h,
            "T_max_0": climatic_stads['T_max_0'], "T_min_0": climatic_stads['T_min_0'], 
            "RH_0": climatic_stads['RH_0'], "raining_total_0": climatic_stads['raining_total_0'], 
            "wind_avg_0": climatic_stads['wind_avg_0'], "wind_max_0": climatic_stads['wind_max_0'], 
            "total_sky_cover_0": climatic_stads['total_sky_cover_0'], #average parameters of the actual day
            "T_max_1": climatic_stads['T_max_1'], "T_min_1": climatic_stads['T_min_1'], 
            "RH_1": climatic_stads['RH_1'], "raining_total_1": climatic_stads['raining_total_1'], 
            "wind_avg_1": climatic_stads['wind_avg_1'], "wind_max_1": climatic_stads['wind_max_1'], 
            "total_sky_cover_1": climatic_stads['total_sky_cover_1'], #average parameters of the day 1 after
            "T_max_2": climatic_stads['T_max_2'], "T_min_2": climatic_stads['T_min_2'], 
            "RH_2": climatic_stads['RH_2'], "raining_total_2": climatic_stads['raining_total_2'], 
            "wind_avg_2": climatic_stads['wind_avg_2'], "wind_max_2": climatic_stads['wind_max_2'], 
            "total_sky_cover_2": climatic_stads['total_sky_cover_2'], #average parameters of the day 2 after
            "PMV": PMV
            }
        )
        self.obs_queue.put(self.next_obs)
    
    def _send_actions(self, state_argument):
        """
        EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        if self.act_queue.empty():
            return
        next_action = self.act_queue.get()
        assert isinstance(next_action, float)

        descentralized_action = transform_centralized_action(next_action)
        DSP_H = descentralized_action[0]
        DSP_C = descentralized_action[1]
        a_tp1_vn = descentralized_action[2]
        a_tp1_vs = descentralized_action[3]
        a_tp1_p = descentralized_action[4]
        
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["Heating"],
            actuator_value=DSP_H
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["Cooling"],
            actuator_value=DSP_C
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["VentN"],
            actuator_value=a_tp1_vn
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["VentS"],
            actuator_value=a_tp1_vs
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["Shading"],
            actuator_value=a_tp1_p
        )

    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) \
            and not self.x.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        """initialize sensors/actuators handles to interact with during simulation"""
        if not self.initialized:
            if not self.x.api_data_fully_ready(state_argument):
                return False

            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                self.var_handles,
                self.meter_handles,
                self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode('utf-8')
                    print(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                    )
                    exit(1)

            self.init_queue.put("")
            self.initialized = True

        return True

    def _flush_queues(self):
        for q in [self.obs_queue, self.act_queue]:
            while not q.empty():
                q.get()

class EnergyPlusEnv_v0(gym.Env):

    def __init__(self, env_config: Dict[str, Any], beta: float):
        self.env_config = env_config
        self.episode = -1
        self.timestep = 0
        self.beta = beta

        # observation space:        
        # "Bw","To","Ti","v","d","RHi","dh","dc","rad","To_p1h","To_p2h","To_p3h"
        # "T_max_0","T_min_0","RH_0","raining_total_0","wind_avg_0","wind_max_0",
        # "total_sky_cover_0","T_max_1","T_min_1","RH_1","raining_total_1","wind_avg_1",
        # "wind_max_1","total_sky_cover_1","T_max_2","T_min_2","RH_2","raining_total_2",
        # "wind_avg_2","wind_max_2","total_sky_cover_2"
            
        self.observation_space = gym.spaces.Box(float("-inf"), float("inf"), (33,))
        self.last_obs = {}

        # action space: supply air temperature (100 possible values)
        self.action_space: Discrete = Discrete(528)

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        self.episode += 1
        self.last_obs = self.observation_space.sample()

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue
        )
        self.energyplus_runner.start()

        # wait for E+ warmup to complete
        if not self.energyplus_runner.initialized:
            self.energyplus_runner.init_queue.get()

        try:
            obs = self.obs_queue.get()
        except Empty:
            obs = self.last_obs

        return np.array(list(obs.values())), {}

    def step(self, action):
        self.timestep += 1
        done = False

        # check for simulation errors
        if self.energyplus_runner.failed():
            print(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")
            exit(1)

        # rescale agent decision to actuator range
        sat_spt_value = self._rescale(
            n=int(action),  # noqa
            range1=(0, self.action_space.n),
            range2=(15, 30)
        )

        # enqueue action (received by EnergyPlus through dedicated callback)
        # then wait to get next observation.
        # timeout is set to 2s to handle start and end of simulation cases, which happens async
        # and materializes by worker thread waiting on this queue (EnergyPlus callback
        # not consuming yet/anymore)
        # timeout value can be increased if E+ warmup period is longer or if step takes longer
        timeout = 2
        try:
            self.act_queue.put(sat_spt_value, timeout=timeout)
            self.last_obs = obs = self.obs_queue.get(timeout=timeout)
        except (Full, Empty):
            done = True
            obs = self.last_obs

        # this won't always work (reason for queue timeout), as simulation
        # sometimes ends with last reported progress at 99%.
        if self.energyplus_runner.progress_value == 100:
            print("reached end of simulation")
            done = True

        # compute reward
        reward = self._compute_reward(obs, self.beta)

        obs_vec = np.array(list(obs.values()))
        return obs_vec, reward, done, False, {}

    def render(self, mode="human"):
        pass

    @staticmethod
    def _compute_reward(obs: Dict[str, float], beta) -> float:
        e_C = (abs(obs["Cooling"]))/(3.6*1000000) # The energy consumption e is equal to the q_supp value but in kWh not in J
        e_H = (abs(obs["Heating"]))/(3.6*1000000) # The energy consumption e is equal to the q_supp value but in kWh not in J
        
        reward = -beta*(e_H + e_C)/(0.4/6) - (1 - beta)*((abs(obs["PMV"])**2)/9)
        
        return reward
