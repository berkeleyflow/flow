import logging
from flow.utils.flow_warnings import deprecation_warning


class SumoParams:

    def __init__(self,
                 port=None,
                 sim_step=0.1,
                 emission_path=None,
                 lateral_resolution=None,
                 no_step_log=True,
                 sumo_binary="sumo",
                 overtake_right=False,
                 ballistic=False,
                 seed=None,
                 restart_instance=False,
                 print_warnings=True,
                 teleport_time=-1):
        """Sumo-specific parameters

        These parameters are used to customize a sumo simulation instance upon
        initialization. This includes passing the simulation step length,
        specifying whether to use sumo's gui during a run, and other features
        described in the Attributes below.

        Attributes
        ----------
        port: int, optional
            Port for Traci to connect to; finds an empty port by default
        sim_step: float optional
            seconds per simulation step; 0.1 by default
        emission_path: str, optional
            Path to the folder in which to create the emissions output.
            Emissions output is not generated if this value is not specified
        lateral_resolution: float, optional
            width of the divided sublanes within a lane, defaults to None (i.e.
            no sublanes). If this value is specified, the vehicle in the
            network cannot use the "LC2013" lane change model.
        no_step_log: bool, optional
            specifies whether to add sumo's step logs to the log file, and
            print them into the terminal during runtime, defaults to True
        sumo_binary: str, optional
            specifies whether to visualize the rollout(s). May be:
                - 'sumo-gui' to run the experiment with the gui
                - 'sumo' to run without the gui (default)
        overtake_right: bool, optional
            whether vehicles are allowed to overtake on the right as well as
            the left
        ballistic: bool, optional
            specifies whether to use ballistic step updates. This is somewhat
            more realistic, but increases the possibility of collisions.
            Defaults to False
        seed: int, optional
            seed for sumo instance
        restart_instance: bool, optional
            specifies whether to restart a sumo instance upon reset. Restarting
            the instance helps avoid slowdowns cause by excessive inflows over
            large experiment runtimes, but also require the gui to be started
            after every reset if "sumo_binary" is set to True.
        print_warnings: bool, optional
            If set to false, this will silence sumo warnings on the stdout
        teleport_time: int, optional
            If negative, vehicles don't teleport in gridlock. If positive,
            they teleport after teleport_time seconds

        """
        self.port = port
        self.sim_step = sim_step
        self.emission_path = emission_path
        self.lateral_resolution = lateral_resolution
        self.no_step_log = no_step_log
        self.sumo_binary = sumo_binary
        self.seed = seed
        self.ballistic = ballistic
        self.overtake_right = overtake_right
        self.restart_instance = restart_instance
        self.print_warnings = print_warnings
        self.teleport_time = teleport_time


class EnvParams:

    def __init__(self,
                 vehicle_arrangement_shuffle=False,
                 starting_position_shuffle=False,
                 additional_params=None,
                 horizon=500,
                 sort_vehicles=False,
                 warmup_steps=0,
                 sims_per_step=1,
                 evaluate=False):
        """Environment and experiment-specific parameters.

        This includes specifying the bounds of the action space and relevant
        coefficients to the reward function, as well as specifying how the
        positions of vehicles are modified in between rollouts.

        Attributes
            vehicle_arrangement_shuffle: bool, optional
                determines if initial conditions of vehicles are shuffled at
                reset; False by default
            starting_position_shuffle: bool, optional
                determines if starting position of vehicles should be updated
                between rollouts; False by default
            additional_params: dict, optional
                Specify additional environment params for a specific
                environment configuration
            horizon: int, optional
                number of steps per rollouts
            sort_vehicles: bool, optional
                specifies whether vehicles are to be sorted by position during
                a simulation step. If set to True, the environment parameter
                self.sorted_ids will return a list of all vehicles ideas sorted
                by their absolute position.
            warmup_steps: int, optional
                number of steps performed before the initialization of training
                during a rollout. These warmup steps are not added as steps
                into training, and the actions of rl agents during these steps
                are dictated by sumo. Defaults to zero
            sims_per_step: int, optional
                number of sumo simulation steps performed in any given rollout
                step. RL agents perform the same action for the duration of
                these simulation steps.
            evaluate: bool, optional
                flag indicating that the evaluation reward should be used
                so the evaluation reward should be used rather than the
                normal reward

        """
        self.vehicle_arrangement_shuffle = vehicle_arrangement_shuffle
        self.starting_position_shuffle = starting_position_shuffle
        self.additional_params = \
            additional_params if additional_params is not None else {}
        self.horizon = horizon
        self.sort_vehicles = sort_vehicles
        self.warmup_steps = warmup_steps
        self.sims_per_step = sims_per_step
        self.evaluate = evaluate

    def get_additional_param(self, key):
        return self.additional_params[key]


class NetParams:

    def __init__(self,
                 no_internal_links=True,
                 in_flows=None,
                 osm_path=None,
                 netfile=None,
                 additional_params=None):
        """Network configuration parameters

        Unlike most other parameters, NetParams may vary drastically dependent
        on the specific network configuration. For example, for the ring road
        the network parameters will include a characteristic length, number of
        lanes, and speed limit.

        In order to determine which additional_params variable may be needed
        for a specific scenario, refer to the ADDITIONAL_NET_PARAMS variable
        located in the scenario file.

        Parameters
        ----------
        no_internal_links : bool, optional
            determines whether the space between edges is finite. Important
            when using networks with intersections; default is False
        in_flows : InFlows type, optional
            specifies the inflows of specific edges and the types of vehicles
            entering the network from these edges
        osm_path : str, optional
            path to the .osm file that should be used to generate the network
            configuration files. This parameter is only needed / used if the
            OpenStreetMapGenerator generator class is used.
        netfile : str, optional
            path to the .net.xml file that should be passed to SUMO. This is
            only needed / used if the NetFileGenerator class is used, such as
            in the case of Bay Bridge experiments (which use a custom net.xml
            file)
        additional_params : dict, optional
            network specific parameters; see each subclass for a description of
            what is needed
        """
        self.no_internal_links = no_internal_links
        self.in_flows = in_flows
        self.osm_path = osm_path
        self.netfile = netfile
        self.additional_params = additional_params or {}


class InitialConfig:

    def __init__(self,
                 shuffle=False,
                 spacing="uniform",
                 min_gap=0,
                 perturbation=0.0,
                 x0=0,
                 bunching=0,
                 lanes_distribution=float("inf"),
                 edges_distribution="all",
                 positions=None,
                 lanes=None,
                 additional_params=None):
        """Initial configuration parameters.

        These parameters that affect the positioning of vehicle in the
        network at the start of a rollout. By default, vehicles are uniformly
        distributed in the network.

        Attributes
        ----------
        shuffle: bool, optional
            specifies whether the ordering of vehicles in the Vehicles class
            should be shuffled upon initialization.
        spacing: str, optional
            specifies the positioning of vehicles in the network relative to
            one another. May be one of: "uniform", "random", or "custom".
            Default is "uniform".
        min_gap: float, optional
            minimum gap between two vehicles upon initialization, in meters.
            Default is 0 m.
        x0: float, optional
            position of the first vehicle to be placed in the network
        perturbation: float, optional
            standard deviation used to perturb vehicles from their uniform
            position, in meters. Default is 0 m.
        bunching: float, optional
            reduces the portion of the network that should be filled with
            vehicles by this amount.
        lanes_distribution: int, optional
            number of lanes vehicles should be dispersed into. If the value is
            greater than the total number of lanes on an edge, vehicles are
            spread across all lanes.
        edges_distribution: list <str>, optional
            list of edges vehicles may be placed on initialization, default is
            all lanes (stated as "all")
        positions: list, optional
            used if the user would like to specify user-generated initial
            positions.
        lanes: list, optional
            used if the user would like to specify user-generated initial
            positions.
        additional_params: dict, optional
            some other network-specific params
        """
        self.shuffle = shuffle
        self.spacing = spacing
        self.min_gap = min_gap
        self.perturbation = perturbation
        self.x0 = x0
        self.bunching = bunching
        self.lanes_distribution = lanes_distribution
        self.edges_distribution = edges_distribution
        self.positions = positions
        self.lanes = lanes
        self.additional_params = additional_params or dict()

    def get_additional_params(self, key):
        return self.additional_params[key]


class SumoCarFollowingParams:

    def __init__(self,
                 accel=1.0,
                 decel=1.5,
                 sigma=0.5,
                 tau=1.0,  # past 1 at sim_step=0.1 you no longer see waves
                 min_gap=2.5,
                 max_speed=30,
                 speed_factor=1.0,
                 speed_dev=0.1,
                 impatience=0.5,
                 car_follow_model="IDM",
                 **kwargs):
        """Parameters for sumo-controlled acceleration behavior

        Attributes
        ----------
        accel: float
            see Note
        decel: float
            see Note
        sigma: float
            see Note
        tau: float
            see Note
        min_gap: float
            see minGap Note
        max_speed: float
            see maxSpeed Note
        speed_factor: float
            see speedFactor Note
        speed_dev: float
            see speedDev in Note
        impatience: float
            see Note
        car_follow_model: str
            see carFollowModel in Note
        kwargs: dict
            used to handle deprecations

        Note
        ----
        For a description of all params, see:
        http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes

        """
        # check for deprecations (minGap)
        if "minGap" in kwargs:
            deprecation_warning(self, "minGap", "min_gap")
            min_gap = kwargs["minGap"]

        # check for deprecations (maxSpeed)
        if "maxSpeed" in kwargs:
            deprecation_warning(self, "maxSpeed", "max_speed")
            max_speed = kwargs["maxSpeed"]

        # check for deprecations (speedFactor)
        if "speedFactor" in kwargs:
            deprecation_warning(self, "speedFactor", "speed_factor")
            speed_factor = kwargs["speedFactor"]

        # check for deprecations (speedDev)
        if "speedDev" in kwargs:
            deprecation_warning(self, "speedDev", "speed_dev")
            speed_dev = kwargs["speedDev"]

        # check for deprecations (carFollowModel)
        if "carFollowModel" in kwargs:
            deprecation_warning(self, "carFollowModel", "car_follow_model")
            car_follow_model = kwargs["carFollowModel"]

        # create a controller_params dict with all the specified parameters
        self.controller_params = {
            "accel": accel,
            "decel": decel,
            "sigma": sigma,
            "tau": tau,
            "minGap": min_gap,
            "maxSpeed": max_speed,
            "speedFactor": speed_factor,
            "speedDev": speed_dev,
            "impatience": impatience,
            "carFollowModel": car_follow_model,
        }


class SumoLaneChangeParams:

    def __init__(self,
                 model="LC2013",
                 lc_strategic=1.0,
                 lc_cooperative=1.0,
                 lc_speed_gain=1.0,
                 lc_keep_right=1.0,
                 lc_look_ahead_left=2.0,
                 lc_speed_gain_right=1.0,
                 lc_sublane=1.0,
                 lc_pushy=0,
                 lc_pushy_gap=0.6,
                 lc_assertive=1,
                 lc_impatience=0,
                 lc_time_to_impatience=float("inf"),
                 lc_accel_lat=1.0,
                 **kwargs):
        """Parameters for sumo-controlled lane change behavior

        Attributes
        ----------
        model: str, optional
            see laneChangeModel in Note
        lc_strategic: float, optional
            see lcStrategic in Note
        lc_cooperative: float, optional
            see lcCooperative in Note
        lc_speed_gain: float, optional
            see lcSpeedGain in Note
        lc_keep_right: float, optional
            see lcKeepRight in Note
        lc_look_ahead_left: float, optional
            see lcLookaheadLeft in Note
        lc_speed_gain_right: float, optional
            see lcSpeedGainRight in Note
        lc_sublane: float, optional
            see lcSublane in Note
        lc_pushy: float, optional
            see lcPushy in Note
        lc_pushy_gap: float, optional
            see lcPushyGap in Note
        lc_assertive: float, optional
            see lcAssertive in Note
        lc_impatience: float, optional
            see lcImpatience in Note
        lc_time_to_impatience: float, optional
            see lcTimeToImpatience in Note
        lc_accel_lat: float, optional
            see lcAccelLate in Note
        kwargs: dict
            used to handle deprecations

        Note
        ----
        For a description of all params, see:
        http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes

        """
        # check for deprecations (lcStrategic)
        if "lcStrategic" in kwargs:
            deprecation_warning(self, "lcStrategic", "lc_strategic")
            lc_strategic = kwargs["lcStrategic"]

        # check for deprecations (lcCooperative)
        if "lcCooperative" in kwargs:
            deprecation_warning(self, "lcCooperative", "lc_cooperative")
            lc_cooperative = kwargs["lcCooperative"]

        # check for deprecations (lcSpeedGain)
        if "lcSpeedGain" in kwargs:
            deprecation_warning(self, "lcSpeedGain", "lc_speed_gain")
            lc_speed_gain = kwargs["lcSpeedGain"]

        # check for deprecations (lcKeepRight)
        if "lcKeepRight" in kwargs:
            deprecation_warning(self, "lcKeepRight", "lc_keep_right")
            lc_keep_right = kwargs["lcKeepRight"]

        # check for deprecations (lcLookaheadLeft)
        if "lcLookaheadLeft" in kwargs:
            deprecation_warning(self, "lcLookaheadLeft", "lc_look_ahead_left")
            lc_look_ahead_left = kwargs["lcLookaheadLeft"]

        # check for deprecations (lcSpeedGainRight)
        if "lcSpeedGainRight" in kwargs:
            deprecation_warning(
                self, "lcSpeedGainRight", "lc_speed_gain_right")
            lc_speed_gain_right = kwargs["lcSpeedGainRight"]

        # check for deprecations (lcSublane)
        if "lcSublane" in kwargs:
            deprecation_warning(self, "lcSublane", "lc_sublane")
            lc_sublane = kwargs["lcSublane"]

        # check for deprecations (lcPushy)
        if "lcPushy" in kwargs:
            deprecation_warning(self, "lcPushy", "lc_pushy")
            lc_pushy = kwargs["lcPushy"]

        # check for deprecations (lcPushyGap)
        if "lcPushyGap" in kwargs:
            deprecation_warning(self, "lcPushyGap", "lc_pushy_gap")
            lc_pushy_gap = kwargs["lcPushyGap"]

        # check for deprecations (lcAssertive)
        if "lcAssertive" in kwargs:
            deprecation_warning(self, "lcAssertive", "lc_assertive")
            lc_assertive = kwargs["lcAssertive"]

        # check for deprecations (lcImpatience)
        if "lcImpatience" in kwargs:
            deprecation_warning(self, "lcImpatience", "lc_impatience")
            lc_impatience = kwargs["lcImpatience"]

        # check for deprecations (lcTimeToImpatience)
        if "lcTimeToImpatience" in kwargs:
            deprecation_warning(
                self, "lcTimeToImpatience", "lc_time_to_impatience")
            lc_time_to_impatience = kwargs["lcTimeToImpatience"]

        # check for deprecations (lcAccelLat)
        if "lcAccelLat" in kwargs:
            deprecation_warning(self, "lcAccelLat", "lc_accel_lat")
            lc_accel_lat = kwargs["lcAccelLat"]

        # check for valid model
        if model not in ["LC2013", "SL2015"]:
            logging.error("Invalid lane change model! Defaulting to LC2013")
            model = "LC2013"

        if model == "LC2013":
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lc_strategic),
                "lcCooperative": str(lc_cooperative),
                "lcSpeedGain": str(lc_speed_gain),
                "lcKeepRight": str(lc_keep_right),
                # "lcLookaheadLeft": str(lcLookaheadLeft),
                # "lcSpeedGainRight": str(lcSpeedGainRight)
            }
        elif model == "SL2015":
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lc_strategic),
                "lcCooperative": str(lc_cooperative),
                "lcSpeedGain": str(lc_speed_gain),
                "lcKeepRight": str(lc_keep_right),
                "lcLookaheadLeft": str(lc_look_ahead_left),
                "lcSpeedGainRight": str(lc_speed_gain_right),
                "lcSublane": str(lc_sublane),
                "lcPushy": str(lc_pushy),
                "lcPushyGap": str(lc_pushy_gap),
                "lcAssertive": str(lc_assertive),
                "lcImpatience": str(lc_impatience),
                "lcTimeToImpatience": str(lc_time_to_impatience),
                "lcAccelLat": str(lc_accel_lat)
            }


class InFlows:

    def __init__(self):
        """
        Used to add inflows to a network. Inflows can be specified for any edge
        that has a specified route or routes.
        """
        self.num_flows = 0
        self.__flows = []

    def add(self,
            veh_type,
            edge,
            begin=1,
            end=2e6,
            vehs_per_hour=None,
            period=None,
            probability=None,
            number=None,
            **kwargs):
        """Specifies a new inflow for a given type of vehicles and edge.

        Parameters
        ----------
        veh_type: str
            type of vehicles entering the edge, must match one of the types set
            in the Vehicles class.
        edge: str
            starting edge for vehicles in this inflow.
        begin: float, optional
            see Note
        end: float, optional
            see Note
        vehs_per_hour: float, optional
            see vehsPerHour in Note
        period: float, optional
            see Note
        probability: float, optional
            see Note
        number: int, optional
            see Note
        kwargs: dict, optional
            see Note

        Note
        ----
        For information on the parameters start, end, vehs_per_hour, period,
        probability, number, as well as other vehicle type and routing
        parameters that may be added via \*\*kwargs, refer to:
        http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes

        """
        # check for deprecations (vehsPerHour)
        if "vehsPerHour" in kwargs:
            deprecation_warning(self, "vehsPerHour", "vehs_per_hour")
            vehs_per_hour = kwargs["vehsPerHour"]
            # delete since all parameters in kwargs are used again later
            del kwargs["vehsPerHour"]

        new_inflow = {"name": "flow_%d" % self.num_flows, "vtype": veh_type,
                      "route": "route" + edge, "end": end}

        new_inflow.update(kwargs)

        if begin is not None:
            new_inflow["begin"] = begin
        if vehs_per_hour is not None:
            new_inflow["vehsPerHour"] = vehs_per_hour
        if period is not None:
            new_inflow["period"] = period
        if probability is not None:
            new_inflow["probability"] = probability
        if number is not None:
            new_inflow["number"] = number

        self.__flows.append(new_inflow)

        self.num_flows += 1

    def get(self):
        return self.__flows
