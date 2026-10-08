#from sentence_transformers import SentenceTransformer
from collections import Counter
import os
from itertools import combinations
from pathlib import Path

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from pathlib import Path
import japanize_matplotlib
import numpy as np
from tqdm import tqdm
import asyncio
from googletrans import Translator
from copy import deepcopy
import networkx as nx
from networkx.algorithms import community

import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from pathlib import Path

# --- macOS Japanese font setup ---
jp_font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"

if Path(jp_font_path).exists():
    font_manager.fontManager.addfont(jp_font_path)
    rcParams["font.family"] = "Hiragino Sans"
else:
    print("Japanese font not found; labels may not render correctly.")

# Avoid minus sign issues
rcParams["axes.unicode_minus"] = False



# party dictionary
with open("nd_groups_translation.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

party_dict = {}
for line in lines:
    line = line.strip()
    if not line or ":" not in line:  # skip empty or malformed lines
        continue
    key, value = line.split(":", 1)  # split only on the first colon
    party_dict[key.strip()] = value.strip()

class SpeechAnalyzer:
    def __init__(self, speech_df, stopwords):
        self.speech_df = speech_df
        #self.model_transformers = SentenceTransformer('sentence-transformers/LaBSE')
        self.nlp = spacy.load("ja_core_news_lg")
        self.stopwords = stopwords
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words=self.stopwords)
        self.feature_names = None
        self.X = None# vectorized corpus
        self.corpus = list(speech_df['comment-body'])
        self.tokenized_texts = self._pretokenize_texts()
        self.W = None
        self.H = None
        self.extended_df = None
    





    def _pretokenize_texts(self):
        print("Starting initial tokenization...")
        pretokenized_texts = []
        for text in tqdm(list(self.speech_df['comment-body'])):
            pretokenized_texts.append(self._word_tokenize(text))
        return pretokenized_texts
        


    def _sentence_tokenize(self, text):
        "tokenizes text into sentences"
        doc = self.nlp(text)
        tokenized_sentences = [sent.text for sent in doc.sents]
        return tokenized_sentences
    
    def _word_tokenize(self, text):
        "tokenizes and filters stopwords"
        word_tokens = []
        if len(text) > 1500:
            subtexts = text.split('。')
            for subtext in subtexts:
                interim_doc = self.nlp(subtext)
                interim_tokens = [token.lemma_ for token in interim_doc]
                word_tokens += interim_tokens
        else:
            doc = self.nlp(text)
            word_tokens = [token.lemma_ for token in doc]
        return word_tokens
    

    def calculate_vectors(self, way="tf-idf"):
        """represents everu speech in vector form for further topic modeling"""
        corpus = [" ".join(text) for text in self.tokenized_texts]
        if way == "tf-idf":
            vectors = self.vectorizer.fit_transform(corpus)
            self.X = vectors
            return vectors
        

    def test_coherence(self, min_topics=3, max_topics=15, top_n_words=10):
        if self.X == None:
            self.calculate_vectors()
        print("Vectorizing...")
        feature_names = self.vectorizer.get_feature_names_out()
        dictionary = corpora.Dictionary(self.tokenized_texts)
        bow_corpus = [dictionary.doc2bow(text) for text in self.tokenized_texts]

        coherence_scores = []
        topic_range = range(min_topics, max_topics + 1)
        for n_topics in tqdm(topic_range):
            nmf = NMF(n_components=n_topics, random_state=42)
            W = nmf.fit_transform(self.X)
            H = nmf.components_

            topics = []
            for topic in H:
                top_words = [feature_names[i] for i in topic.argsort()[::-1][:top_n_words]]
                topics.append(top_words)

            coherence_model = CoherenceModel(
                topics=topics,
                texts=self.tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
                        )
            score = coherence_model.get_coherence()
            coherence_scores.append(score)
            print(f"Topics: {n_topics}, Coherence (c_v): {score:.4f}")


        plt.figure()
        plt.plot(topic_range, coherence_scores, marker='o')
        plt.title("Topic Coherence vs Number of Topics (NMF + TF-IDF)")
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score (c_v)")
        plt.xticks(topic_range)
        plt.grid(True)
        plt.show()


    def extract_topics(self, n_topics = 10, top_n_words=10, model='nmf', print_topics = False, translation = True):
        if model == 'nmf':
            print("Vectorizing texts...")

            if self.X == None:
                self.calculate_vectors()

            X = self.X
            nmf = NMF(n_components=n_topics, random_state=42)
            print("Computing topic matrix...")
            W = nmf.fit_transform(X)  # document-topic matrix
            self.W = W
            H = nmf.components_       # topic-word matrix
            self.H = H

            #top_words =10
            #top_n_words =10

            # 5. Display top words per topic
            feature_names = self.vectorizer.get_feature_names_out()
            self.feature_names = feature_names
            output_topics = ''
            for idx, topic in enumerate(H):
                words = [feature_names[i] for i in topic.argsort()[::-1][:top_n_words]]
                if print_topics:
                    print(f"Topic {idx}: {' | '.join(words)}")
                output_topics += f"Topic {idx}: {' | '.join(words)}\n"

            for idx, topic in enumerate(H):
                top_indices = topic.argsort()[::-1][:top_n_words]
                words_weights = [(feature_names[i], round(topic[i], 4)) for i in top_indices]
                if print_topics:
                    print(f"Topic {idx}: {words_weights}")

            translated_topics = None
            if translation:
                translator = Translator()
                translated_topics = translator.translate(output_topics, src='ja', dest='en')


            return {
        "topic_word_matrix": H,
        "document_topic_matrix": W,
        "feature_names": feature_names,
        "topics_ja": output_topics,
        "topics_en": translated_topics
        }
        

    def calculate_share(self):
        topic_share_df = pd.DataFrame(self.W, columns=[f"Topic_{i}" for i in range(self.W.shape[1])])
        self.extended_df = pd.concat([self.speech_df, topic_share_df], axis=1)
        #print(self.extended_df)
        return self.extended_df

        



class SpeechCharacteristics:
    def __init__(self, input_df, start_year = 1949, end_year = 2025):
        self.start_year = start_year
        self.end_year = end_year
        self.input_df = input_df
        self.target_df = self._split_dataset()
        self.topic_extractor = None
        self.extended_df = None
        
        

    def _split_dataset(self):
        # Ensure the date column is in datetime format
        self.input_df['date'] = pd.to_datetime(self.input_df['date'], errors='coerce')

        # Extract the year from the date
        self.input_df['year'] = self.input_df['date'].dt.year

        # Filter by the interval [start_year, end_year]
        target_df = self.input_df[
            (self.input_df['year'] >= self.start_year) & 
            (self.input_df['year'] <= self.end_year)
        ]
        return target_df.reset_index(drop=True)
    
    def get_speakers(self):
        """returns the speakers and their counts"""
        speakers = Counter(list(self.target_df['speaker']))
        speakers_df = pd.DataFrame(speakers.items(), columns=['Speaker', 'Count'])
        return speakers_df
    
    def get_share_of_guest_speakers(self, picture=False):
        self.target_df['speaker_role'] = self.target_df['speaker_role'].fillna('国会議員')
        speakers_guest_status = list(self.target_df['speaker_role'])
        role_counts = Counter(speakers_guest_status)
        picture_df = pd.DataFrame(role_counts.items(), columns=['Role', 'Count'])
        picture_df = picture_df.sort_values(by='Count', ascending=False)
        replace_roles = {'国会議員': "Diet Member", '参考人': "Expert Witness",
            "公述人": 'Public Commenter', "証人": 'Witness (under oath)'}
        picture_df['Role'] = picture_df['Role'].replace(replace_roles)

        plt.figure()
        plt.margins(x=0,y=0)
        plt.bar(picture_df['Role'], picture_df['Count'])
        plt.title('Distribution of Speakers by Role')
        plt.xlabel('Role')
        plt.ylabel('Number')
        plt.xticks(rotation=90)
        plt.yticks()
        plt.tight_layout()
        plt.show()
        return picture_df
    
    def get_party_shares(self, picture=False, length=10, width=10):
        party_membership = self.target_df["speaker_group"]
        party_membership = party_membership.dropna()
        party_counts = Counter(list(party_membership))
        picture_df = pd.DataFrame(party_counts.items(), columns=['Party', 'Count'])
        picture_df['Party'] = picture_df['Party'].replace(party_dict)
        plt.figure(figsize=(length, width))
        plt.margins(x=0,y=0)
        plt.bar(picture_df['Party'], picture_df['Count'])
        plt.title('Party Membership of ND Speakers')
        plt.xlabel('Party')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.yticks()
        plt.tight_layout()
        plt.show()
        return picture_df
    
    def test_topics(self, stopwords):
        if self.topic_extractor == None:
            self.topic_extractor = SpeechAnalyzer(self.target_df,stopwords)
        

        
    
    def extract_topics(self, stopwords, topic_min=3, topic_max=10, coherence=True, save_topics = False):
        if self.topic_extractor == None:
            self.topic_extractor = SpeechAnalyzer(self.target_df,stopwords)
        if coherence:
            self.topic_extractor.test_coherence(topic_min, topic_max)
        for topic_n in tqdm(range(topic_min, topic_max+1)):
            topics = self.topic_extractor.extract_topics(topic_n)
            output_topics = f"{topics['topics_ja']}\n\n{topics['topics_en']}"
            if save_topics:
                directory = f"{self.start_year}-{self.end_year}"
                os.makedirs(directory, exist_ok=True)
                with open(f"{directory}/{topic_n}.txt", 'w', encoding='utf-8') as f:
                    f.write(output_topics)
            else:
                print(output_topics)


    def get_topic_distribution(self, treshhold_sensetivity = 0.1, rename_topics = False, select_topic = False, figure_type = False, figsize = (10,10)):
        """
        treshhold_sensetivity: what should be the minimum share of topic to be counted 
        rename_topics: list of topics to rename
        
        
        """
        # here also calculate weighted topic influence
        "run after extract_topics"
        "treshhold sensetivity: minimal share of topic to be him considered in counting"
        if self.topic_extractor == None:
            print('You shall perform topic modeling first.\n Run .extract_topics() first')
            return ValueError
        extended_df = self.topic_extractor.calculate_share()
        df_columns = list(extended_df.columns)
        topic_columns = [col for col in df_columns if "Topic" in col]
        rename_col_dict = {}
        if rename_topics:
            for i,t in enumerate(rename_topics):
                rename_col_dict[topic_columns[i]] = t

        self.extended_df = extended_df.copy()
        self.extended_df = extended_df.rename(columns=rename_col_dict)

        total_dominant_count = {} # here we count all topics
        weights_total_count = {} # here sum weights
        for topic in topic_columns:
            total_dominant_count[topic] = 0
            weights_total_count[topic] = 0
        yearly_count = {}
        yearly_weights = {}
        for year in range(self.start_year, self.end_year+1):
            yearly_count[year] = deepcopy(total_dominant_count)
            yearly_weights[year] = deepcopy(total_dominant_count)

        for i in range(len(extended_df)):
            row = extended_df.iloc[i]
            dominant_topic = row[topic_columns].idxmax()
            max_value = row[topic_columns].max()
            if max_value >= treshhold_sensetivity:
                total_dominant_count[dominant_topic] += 1
                year = row['year']
                yearly_count[year][dominant_topic] += 1
            for col in topic_columns:
                weights_total_count[col] += row[col]
                yearly_weights[year][col] += row[col]

        graph_0 = pd.DataFrame.from_dict(total_dominant_count, orient='index')
        graph_0.index.name = 'Topic'
        graph_1 = pd.DataFrame.from_dict(yearly_count, orient='index')
        graph_1.index.name = 'year'
        graph_1 = graph_1.sort_index()
        graph_2 =pd.DataFrame.from_dict(weights_total_count, orient='index')
        graph_2.index.name = 'Topic'
        graph_3 = pd.DataFrame.from_dict(yearly_weights, orient='index')
        graph_3.index.name = 'year'
        graph_3 = graph_3.sort_index()

        if isinstance(rename_topics, list):
            graph_0.index = rename_topics
            graph_1.columns = rename_topics
            graph_2.index = rename_topics
            graph_3.columns = rename_topics

        if select_topic:
            graph_1 = graph_1[select_topic]
            graph_3 = graph_3[select_topic]




        # graph 0
        self._draw_bar_0(graph_0, figsize)

        # graph 1
        self._draw_bar_1(graph_1, figure_type, figsize)

        # graph 2
        self._draw_bar_2(graph_2, figsize)

        #graph 3
        self._draw_bar_3(graph_3, figure_type, figsize)

        return graph_0, graph_1, graph_2, graph_3
    #split this method into separate onesto adjust figure size, add the function to look for a detailed data in particuler year period

    def _draw_bar_0(self, graph, figsize=(10,10)):
        """
        draws the counts of topic appearances

        """
        plt.figure(figsize=(10,10))
        #plt.margins(x=0,y=0)
        graph.plot(kind='bar')
        plt.title(f"Dominant Topic Total Counts, {self.start_year}-{self.end_year}")
        plt.xlabel("Topic")
        plt.ylabel("Count")
        plt.legend().remove()
        plt.tight_layout()
        plt.show()

    def _draw_bar_1(self, graph, figure_type, figsize=(10,10)):
        """
        draws topic count year by year
        """
        plt.figure(figsize=figsize)
        #plt.margins(x=0,y=0)
        graph.plot(kind=figure_type)
        plt.title(f"Topics Appearance Counts, {self.start_year}-{self.end_year}")
        plt.xlabel("Topic")
        plt.ylabel("Counts")
        plt.legend(title = 'Topics')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _draw_bar_2(self,graph, figsize=(10,10)):
        """
        draws accumulated weights for every topic
        """
        plt.figure(figsize=figsize)
        #plt.margins(x=0,y=0)
        graph.plot(kind='bar')
        plt.title(f"Topics Weights Counts, {self.start_year}-{self.end_year}")
        plt.xlabel("Topic")
        plt.ylabel("Weights Accumulated")
        plt.legend(title = 'Topics')
        plt.legend().remove()
        plt.tight_layout()
        plt.show()

    def _draw_bar_3(self, graph, figure_type, figsize=(10,10)):
        """
        calcilates yearly changes of accumulated weights for every topic
        """
        plt.figure(figsize=figsize)
        #plt.margins(x=0,y=0)
        graph.plot(kind=figure_type)
        plt.title(f"Topics Weights Yearly, {self.start_year}-{self.end_year}")
        plt.xlabel("Topic")
        plt.ylabel("Weights Accumulated")
        plt.legend(title = 'Topics')
        plt.tight_layout()
        plt.show()


    
    


    def count_speech_number(self, figsize=(10,10)):
        year_count = []
        for i in range(len(self.target_df)):
            row = self.target_df.iloc[i]
            year_count.append(row['year'])
        years_counted = Counter(year_count)
        picture_df = pd.DataFrame(years_counted.items(), columns=['Year', 'No. of Speeches'])
        plt.figure(figsize=figsize)
        plt.margins(x=0,y=0)
        plt.bar(picture_df['Year'], picture_df['No. of Speeches'])
        plt.title(f'Number of Speeches per Year, {self.start_year}-{self.end_year}')
        plt.xlabel('Year')
        plt.ylabel('No. of Speeches')
        plt.xticks(rotation=90)
        plt.yticks()
        plt.tight_layout()
        plt.show()
        return picture_df




class NetworkAnalyzer:
    def __init__(self, input_df, node_column, interaction_column, start_period, end_period):
        """
        input_df: dataset with ND speeches
        node_column: column name of speakers
        interaction-column: column defining the criteria of interaction either date or session number
        """
        self.initial_df = deepcopy(input_df)
        self.initial_df["date"] = pd.to_datetime(self.initial_df["date"]) # consider that interaction-column is in date format and can be safely converted datatime
        self.node_column = node_column 
        self.interaction_column = interaction_column
        self.start_period = start_period
        self.end_period = end_period
        if self.start_period != None and self.end_period != None:
            self.df = self._choose_date_interval() # folter dataframe for the selected time period
        else: 
            self.df = self.initial_df
        self.weighted_edge_df = None
        self.G = None
        self.all_nodes = None
        self.connected_nodes = None
        self.subgraphs = None
        self.build_graph()


    def _get_expert_nodes(self):
        "helper method to get indices of expert nodes"
        experts_df = self.df[self.df['speaker_role'] == '参考人'].copy()
        experts = experts_df['speaker'].to_list()
        return experts

        
        

        

    def _choose_date_interval(self):
        """ filters dataset in the input interval while class initialization"""
        mask = (self.initial_df["date"] >= self.start_period) & (self.initial_df["date"] <= self.end_period)
        df_filtered = self.initial_df.loc[mask]
        return df_filtered



    def _get_nodes_and_edges(self):
        """extracts nodes and edges from dataset"""
        edges = []
        for interaction, group in self.df.groupby(self.interaction_column):
            nodes = sorted(group[self.node_column].unique())
            for a, b in combinations(nodes, 2):
                edges.append((a, b, interaction))

        return nodes, edges
    
    def _get_graph_df(self):
        """creates weighted dataset with nodes and edges pairs"""
        nodes, edges = self._get_nodes_and_edges()

        edge_df = pd.DataFrame(edges, columns=["source", "target", "date"])

        self.weighted_edge_df = (
            edge_df.groupby(["source", "target"])
            .size()
            .reset_index(name="weight")
        )

    def build_graph(self):
        """builds graph"""
        if self.weighted_edge_df is None:
            self._get_graph_df()

        self.G = nx.from_pandas_edgelist(
            self.weighted_edge_df,
            source="source",
            target="target",
            edge_attr="weight"
        )
    
    def draw_graph(self, fig_size = (10,10), k=0.1, save_pic = False, save_name= 'out_pic.jpg', save_folder=None, subgraph = False,
                   mark_experts = False):
        """
        draws figure
        fig_size = tuple of plt figsize
        k = float adjust the sparsity of the nodes in figure
        mark_experts = marks expert nodes with color
        """
        if subgraph:
            G = subgraph
        else:
            G = self.G

        pos = nx.spring_layout(G, k=k, seed=42)  # smaller k → tighter clusters
        
        if mark_experts:
            expert_nodes = self._get_expert_nodes()
            node_colors = []
            for n in G.nodes():

                if n in expert_nodes:
                    node_colors.append("red")
                else:
                    node_colors.append("lightgray")
        





        plt.figure(figsize=fig_size)
        nx.draw_networkx(G, with_labels=True,
                         pos = pos,  
                         node_size = 100,
                         font_family="Hiragino Sans",
                         font_size = 11,
                         node_color=node_colors if mark_experts else None)

        if save_pic:
            save_folder_path = Path(save_folder) if save_folder else Path.cwd()
            save_path = save_folder_path / save_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Graph saved to: {save_path}")

        plt.show()
        plt.close()
    


    def get_graph_stats(self, save_data=False, save_dir=None, save_prefix=False, subgraph = False, top_n=10):
        """
        calculates graphs parameters
        save_data = Bool, 
        save_path = str, the custom path for file saving
        """

        if subgraph:
            G = subgraph
        else:
            G = self.G
            
        

        gen_stats_df = pd.DataFrame()
        gen_stats_df['Nodes'] = [G.number_of_nodes()]
        gen_stats_df['Edges'] = [G.number_of_edges()]
        gen_stats_df['Density'] = [nx.density(G)]

        edge_stats_df = pd.DataFrame()

        edge_stats_df['Degree Centrality'] = nx.degree_centrality(G)
        edge_stats_df['Betweenness'] = nx.betweenness_centrality(G)
        edge_stats_df['Clustering'] = nx.clustering(G)

        # Girvan–Newman community detection
        communities = community.girvan_newman(G)
        first_level = next(communities)
        for com in list(first_level):
            print(com)
        communities = [list(c) for c in first_level]

        #centrality metrics
        # --- Centrality measures ---
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)
        clos_cent = nx.closeness_centrality(G)
        eig_cent = nx.eigenvector_centrality(G, max_iter=1000)


        # key nodes
        # --- Key nodes ---
        def get_top_nodes(metric_dict, top_n=top_n):
            return sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

        top_nodes = {
            'Degree': get_top_nodes(deg_cent),
            'Betweenness': get_top_nodes(bet_cent),
            'Closeness': get_top_nodes(clos_cent),
            'Eigenvector': get_top_nodes(eig_cent),
    }

        # ratio connected notes vs unconnected nodes
        if not subgraph:
            all_nodes = len(set(self.df[self.interaction_column]))
            interacting_nodes = G.number_of_nodes()
            ratio = interacting_nodes / all_nodes


        # saving
        base_path = Path().cwd()
        if save_prefix == False:
            gen_stats_f_path = Path(f'{self.start_period}-{self.end_period}_gen_stats.csv')
            edge_stats_f_path = Path(f'{self.start_period}-{self.end_period}_edge_stats.csv')
            communities_n_central_f_path = Path(f'{self.start_period}-{self.end_period}_edge_stats.csv')
        else:
            gen_stats_f_path = Path(f'{save_prefix}_{self.start_period}-{self.end_period}_gen_stats.csv')
            edge_stats_f_path = Path(f'{save_prefix}_{self.start_period}-{self.end_period}_edge_stats.csv')
            communities_n_central_f_path = Path(f'{save_prefix}_{self.start_period}-{self.end_period}_communities_and_nodes.txt')

        if save_dir == None:
            gen_stats_full_path = base_path / gen_stats_f_path
            edge_stats_full_path = base_path / edge_stats_f_path
            communities_n_central_full_path = base_path / communities_n_central_f_path
        else:
            save_dir_path = Path(save_dir)
            gen_stats_full_path = base_path / save_dir_path / gen_stats_f_path
            edge_stats_full_path = base_path / save_dir_path / edge_stats_f_path
            communities_n_central_full_path = base_path /save_dir_path / communities_n_central_f_path
            gen_stats_full_path.parent.mkdir(parents=True, exist_ok=True)
            edge_stats_full_path.parent.mkdir(parents=True, exist_ok=True)
            communities_n_central_full_path.parent.mkdir(parents=True, exist_ok=True)


        if save_data == True:
            gen_stats_df.to_csv(gen_stats_full_path, index=False)
            edge_stats_df.to_csv(edge_stats_full_path, index=True)

            with open(communities_n_central_full_path, "w", encoding="utf-8") as f:
                f.write("### Communities Detected ###\n")
                for i, com in enumerate(communities, start=1):
                    f.write(f"\nCommunity {i} ({len(com)} nodes):\n")
                    f.write(", ".join(map(str, com)) + "\n")

                f.write("\n### Top Key Nodes by Centrality ###\n")
                for metric, nodes in top_nodes.items():
                    f.write(f"\n{metric} Centrality (Top {len(nodes)}):\n")
                    for node, score in nodes:
                        f.write(f"  {node}: {score:.4f}\n")

                if not subgraph:
                    f.write(f"\n### Nodes Concentration ###\nConcentration: {round(ratio, 2)}")
            
        
        return gen_stats_df, edge_stats_df, communities, top_nodes
        

    def extract_subgraphs(self, save=False, save_format="gexf", save_dir="subgraphs"):
        """
        Извлекает изолированные подграфы (connected components)
        и сохраняет их как независимые объекты NetworkX Graph.

        save = bool — сохранять ли подграфы на диск
        save_format = 'gexf' | 'graphml' | 'pkl' — формат сохранения
        save_dir = str — папка для сохранения подграфов
        """

        if self.G is None:
            self.build_graph()
        if self.G is None or len(self.G.nodes) == 0:
            print("Empty graph")
            return None

        components = list(nx.connected_components(self.G))
        print(f"Found {len(components)} subgraphs")

        # Словарь подграфов
        self.subgraphs = {}

        for i, comp in enumerate(components, start=1):
            subG = self.G.subgraph(comp).copy()   # копия независимого графа
            self.subgraphs[i] = subG

            # --- Сохранение ---
            if save:
                import os
                os.makedirs(save_dir, exist_ok=True)
                path = f"{save_dir}/subgraph_{i}.{save_format}"

                if save_format == "gexf":
                    nx.write_gexf(subG, path)
                elif save_format == "graphml":
                    nx.write_graphml(subG, path)
                elif save_format == "pkl":
                    import pickle
                    with open(path, "wb") as f:
                        pickle.dump(subG, f)
                else:
                    print(f"Неподдерживаемый формат: {save_format}")
                    continue

        print(f"Subgraphs saved ({len(self.subgraphs)} шт.) в '{save_dir}'")
        return self.subgraphs
    

    def get_subfigure_stats(self, save_subgraphs = False, save_dir=False):
        if self.subgraphs == None:
            self.extract_subgraphs()

        gen_stats_out = []; edge_stats_out = []
        if save_subgraphs == False:
            save_data = False; save_dir = save_dir
        else: 
            save_data = True; save_dir = save_dir

        for i, graph in self.subgraphs.items():
            print(self.subgraphs)
            print(self.subgraphs.items())
            print(i)
            res = self.get_graph_stats(save_data=save_data, save_dir=save_dir, save_prefix=f"subgraph_{i}", subgraph=graph)
            
            gen_stats_out.append(res[0])
            edge_stats_out.append(res[1])

            self.draw_graph(save_pic=True, save_name=f"subgraph_{i}.png", save_folder=save_dir, subgraph=graph)

        return gen_stats_out, edge_stats_out
            


### Replaces names in the dataset ###

def replace_jap_names(df):
    names_df = pd.read_csv('Speakers_Mombetsu_2026.csv', index_col=False)
    names_df = names_df.dropna(subset='English Name')
    mapping = names_df.set_index('Japanese Name')['English Name']
    df['speaker'] = df['speaker'].map(mapping).fillna(df['speaker'])
    return df


