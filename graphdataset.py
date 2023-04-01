import os
from datetime import datetime as dt
import ipaddress

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


class GraphDataset(Dataset):
    labels = ["normal", "anomalous"]
    input_time_format = "%Y-%m-%d %H:%M:%S"
    nx_draw_options = {"node_size": 100, "width": 1, "font_size": 5}
    graph_df_columns = [
        "timeframe",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "protocol",
        "action",
    ]
    features_df_columns = [
        "timeframe",
        "n_unique_ips",
        "n_unique_dst_ips",
        "n_unique_dst_ports",
        "n_accepted",
        "n_dropped",
        "n_connections",
        "n_public",
        "n_private",
        "n_tcp",
        "n_udp",
        "n_icmp",
    ]

    def __init__(self, traffic_file):
        self.traffic_file = traffic_file

        self.graphs = dict()
        self.graph_df = pd.DataFrame(None, columns=self.graph_df_columns)
        self.features_df = pd.DataFrame(None, columns=self.features_df_columns)

        with open(self.traffic_file, "r") as f:
            # while True:
            for i in range(30000):

                line = f.readline()
                if not line:
                    break
                line = line.rstrip()

                self._add_to_graph(line)
                self._add_to_dataframe(line)

        self._generate_features()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.get_label(graph)
        return graph, label

    def _add_to_graph(self, line):
        time = line.split(",")[0]
        time = dt.strptime(time, self.input_time_format)

        timeframe = self._get_timeframe(time, 30)  # 30 seconds
        if timeframe not in self.graphs:
            self.graphs[timeframe] = nx.Graph()

        src_ip = self._get_source_ip(line)
        dst_ip = self._get_destination_ip(line)
        self.graphs[timeframe].add_node(src_ip)
        self.graphs[timeframe].add_node(dst_ip)
        self.graphs[timeframe].add_edge(src_ip, dst_ip)

    def _generate_features(self):
        timeframes = self.graph_df["timeframe"].unique()

        for timeframe in timeframes:
            curr_graph = self.graph_df.loc[self.graph_df["timeframe"] == timeframe]

            # unique ips
            unique_ips = set()
            unique_ips.update(curr_graph["src_ip"].unique())
            unique_ips.update(curr_graph["dst_ip"].unique())
            n_unique_ips = len(unique_ips)

            # unique dst_ips
            unique_dst_ips = set()
            unique_dst_ips.update(curr_graph["dst_ip"].unique())
            n_unique_dst_ips = len(unique_dst_ips)

            # unique dst_ports
            unique_dst_ports = set()
            unique_dst_ports.update(curr_graph["dst_port"].unique())
            n_unique_dst_ports = len(unique_dst_ports)


            # number of accepted connections
            accepted = curr_graph.loc[curr_graph["action"] == "accept"]
            n_accepted = len(accepted)

            # number of dropped connections
            dropped = curr_graph.loc[curr_graph["action"] == "drop"]
            n_dropped = len(dropped)

            # number of dropped connections
            connections = curr_graph
            n_connections = len(connections)

            # number of private/public ips
            public_ips, private_ips = self._split_public_private(unique_ips)
            n_public_ips, n_private_ips = len(public_ips), len(private_ips)

            # number of udp connections
            udps = curr_graph.loc[curr_graph["protocol"] == "udp"]
            n_udps = len(udps)

            # number of tcp connections
            tcps = curr_graph.loc[curr_graph["protocol"] == "tcp"]
            n_tcps = len(tcps)

            # number of icmp connections
            icmps = curr_graph.loc[curr_graph["protocol"] == "icmp"]
            n_icmps = len(icmps)

            row = [
                timeframe,
                n_unique_ips,
                n_unique_dst_ips,
                n_unique_dst_ports,
                n_accepted,
                n_dropped,
                n_connections,
                n_public_ips,
                n_private_ips,
                n_udps,
                n_tcps,
                n_icmps,
            ]
            self.features_df.loc[len(self.features_df.index)] = row

    def _add_to_dataframe(self, line):
        time = line.split(",")[0]
        time = dt.strptime(time, self.input_time_format)

        timeframe = self._get_timeframe(time, 30)  # 30 seconds

        src_ip = self._get_source_ip(line)
        src_port = self._get_source_port(line)
        dst_ip = self._get_destination_ip(line)
        dst_port = self._get_destination_port(line)
        protocol = self._get_protocol(line)
        action = self._get_action(line)

        row = [timeframe, src_ip, src_port, dst_ip, dst_port, protocol, action]
        self.graph_df.loc[len(self.graph_df.index)] = row

    def _get_timeframe(self, time, second_range):
        second_of_day = 0
        second_of_day += time.second
        second_of_day += time.minute * 60
        second_of_day += time.hour * 60 * 60
        return second_of_day % second_range

    def _get_source_ip(self, line):
        return line.split(",")[1]

    def _get_source_port(self, line):
        return line.split(",")[2]

    def _get_destination_ip(self, line):
        return line.split(",")[8]

    def _get_destination_port(self, line):
        return line.split(",")[9]

    def _get_protocol(self, line):
        return line.split(",")[15]

    def _get_action(self, line):
        return line.split(",")[16]

    def _split_public_private(self, ips):
        public, private = [], []
        for ip in ips:
            if ipaddress.ip_address(ip).is_private:
                private.append(ip)
            else:
                public.append(ip)

        return public, private

    def get_label(self, graph):
        # TODO: add logic to label the data
        # right now it returns 'normal' for everything
        return self.labels[0]

    def draw(self, n_graphs=3):

        count = 0
        for time_frame, g in self.graphs.items():
            if count == n_graphs:
                break
            count += 1

            label = self.get_label(g)
            title = f"{time_frame=}, {label=}"

            pos = nx.kamada_kawai_layout(g)
            colors = ["red" for u, v in g.edges()]

            plt.figure(count, figsize=(8, 8))
            nx.draw(g, pos, edge_color=colors, with_labels=True, **self.nx_draw_options)
            plt.title(title)
        plt.show()
