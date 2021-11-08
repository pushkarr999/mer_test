import re

from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import scipy

from sklearn.feature_extraction.text import TfidfVectorizer

expensive_brands = {'AG Adriano Goldschmied', 'ALEX AND ANI', 'ASICS', 'Acacia Swimwear', 'Adrienne Vittadini',
                    'Adidas', 'Affliction', 'Alexander McQueen', 'Alexander Wang', 'Alexis Brittar',
                    'Alice + Olivia', 'AllSaints', 'Almost Famous', 'American Apparel', 'American Eagle', 'Alo',
                    'Anthropologie', 'Arden B', 'Ariat', 'BCBGeneration', 'BEACH RIOT', 'BOYS + ARROWS',
                    'Balenciaga', 'Balmain', 'Banana Republic', 'Beach Bunny', 'Bebe', 'Bed Stu',
                    'Betsey Johnson', 'Birkenstock', 'Botkier', 'Bottega Veneta', 'Brahmin', 'Brandy Melville',
                    'Brighton', 'Brooks', 'Buckle', 'Burberry', 'Burton', 'Calvin Klein', "Candie's", 'Cartier',
                    'Catherine Catherine Malandrino', 'Chaco', 'Chanel', 'Charlotte Russe', 'Chinese Laundry',
                    'Chloé', 'Christian Audigier', 'Christian Louboutin', 'Coach', 'Cole Haan', 'Comme des Garcons',
                    'Converse', 'Customized & Personalized', 'David Yurman', 'Disney', 'Dolce Vita', 'Dooney & Bourke',
                    'Dr. Martens', 'Fashion Nova', 'Fendi', 'Fila', 'Fossil', 'Free People', 'Frye', 'GUESS', 'Gap',
                    'Giuseppe Zanotti', 'Givenchy', 'Gucci', 'Gymshark', 'HERMES', 'Harley-Davidson', 'Hermès',
                    'Herve Leger', 'Hot Topic', 'Hudson Jeans', 'Hunter', 'INC International Concepts', 'Independent',
                    'Infinity', 'Jack Rogers', 'James Avery', 'Jeffery Campbell', 'Jimmy Choo', 'John Hardy',  'Joie',
                    'Jordans', 'Joseph Ribkoff', 'Joules', 'Jovani', 'Judith March', 'Juicy Couture', 'Justin Boots',
                    'Kate Spade', 'Kay Jewelers', 'Kendra Scott', 'L.A.M.B.', 'L.L. Bean', 'LA Hearts', 'LF', 'Lanvin',
                    'Lilly Pulitzer', 'Lilly Pulitzer for Target', 'Longchamp', 'Louis Vuitton', 'LuLaRoe', 'Lucchese',
                    'Lucky Brand', 'Lucy Activewear', 'MARC BY MARC JACOBS', 'MARC JACOBS', 'MCM', 'MIU MIU', 'Maaji',
                    'Madewell', 'Marc New York', 'Masquerade', 'Maurices', 'Merrell', 'Metal Mulisha', 'Michael Kors',
                    'Mikimoto', 'Miss Me', 'Mizuno', 'Modcloth', 'Mori Lee', 'Nasty Gal', 'Native', 'New Balance',
                    'Nfinity', 'Nike', 'Nordstrom', 'PANDORA', 'PINK', 'PUMA', 'Parker', 'Polo Ralph Lauren', 'Prada',
                    'REI', 'Ralph Lauren', 'Rebecca Minkoff', 'Reebok', 'Reformation', 'Rihanna', 'Roberto Cavalli',
                    'Rock Revival', 'Saint Laurent', 'Salvatore Ferragamo', 'Sam Edelman', 'Sbicca', 'Scala',
                    'Sherri Hill', 'Show Me Your MuMu', 'Simply Southern', 'Sky', 'Sorel', 'Sorrelli',
                    'Spell & The Gypsy Collective', 'Stella McCartney', 'Steve Madden', 'Stone Fox Swim',
                    'Stuart Weitzman', 'Swarovski', 'Tanner Mark', 'Terani Couture', 'The North Face', 'Tiffany & Co.',
                    'Timberland', 'Tommy Hilfiger', 'TopShop', 'Tory Burch', 'Tous', 'Triangl', 'Trina Turk',
                    'Tularosa', 'UGG Australia', 'UNIF', 'Under Armour', 'VANS', 'Valentino', 'Van Cleef & Arpels',
                    'Vera Bradley', 'Versace', "Victoria's Secret", 'Vince', 'Vince Camuto', 'Wildfox',
                    'Wildfox Couture', 'Xhilaration', 'YRU', 'YSL Yves Saint Laurent', 'Yeezy', 'Zac Posen',
                    'Zigi Soho', 'adidas Originals', 'lululemon athletica', 'maje', 'rag & bone', 'tokidoki',
                    'vineyard vines'}

def decontract_text(phrase):
    """
    This utility funciton will be used as a part of preprocessing the text.
    It will expand the contracted words. For eg: won't -> will not, I'm -> I am.
    """
    phrase = str(phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def stem_sentence(sentence):
    ps = PorterStemmer()
    words = word_tokenize(sentence)
    root = []
    for w in words:
        root.append(ps.stem(w))
    return " ".join(root)


def preprocess_descriptive_text_column(sentance):
    """
    Description:
    This function will process the text data.
    This function will perform decontracting words, removing stop words, removing special characters and then apply stemming on the words in the sentence.

    Input: original sentence
    Output: processed sentence
    """
    # https://gist.github.com/sebleier/554280
    # we are removing the negative words from the stop words list: 'no', 'nor', 'not', 'shouldn't, won't, etc.
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                 're', 've', 'y']

    sent = decontract_text(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\n', ' ')
    sent = sent.replace('\\"', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)

    root_sent = stem_sentence(sent.lower().strip())
    return root_sent


def fill_missing_values(df):
    """
    Description:
    Filling null values in all columns.

    Input: Dataframe with null values
    Output: Dataframe with no null values
    """
    df['name'].fillna('unk_name', inplace=True)
    df['category_name'].fillna('unk_cat', inplace=True)
    df['brand_name'].fillna('unk_brand', inplace=True)
    df['item_description'].fillna('unk_descr', inplace=True)
    return df


def concat_categories(x):
    return set(x.values)


def brand_guesser(df):
    """
    Description:
    This function is used to guess the missing brand name.
    It will check for an existing brand name mentioned in the item name section.
    We want out guess to be as close to actual as possible, \
    hence we will also check the category of the brand name that is guessed with the already mentioned category of that product.
    If the category matches, then only we will fill that guessed brand name.

    Inputs: dataframe with missing brand names
    Output: dataframe with filled brand names
    """
    existing_brands = df[df['brand_name'] != 'unk_brand']['brand_name'].unique()
    brand_names_categories = dict(
        df[df['brand_name'] != 'unk_brand'][['brand_name', 'category_name']].astype('str').groupby('brand_name').agg(
            concat_categories).reset_index().values.tolist())
    # In the above line, we are creating dictionary of brand name->category, wherever the brand name is missing.
    # This will be helpful to us during guessing the missing brand names.
    filled_brands = []
    for row in tqdm(df[['brand_name', 'name', 'category_name']].values):
        found = False
        if row[0] == 'unk_brand':
            for brand in existing_brands:
                if brand in row[1] and row[2] in brand_names_categories[brand]:
                    filled_brands.append(brand)
                    found = True
                    break
            if not found:
                filled_brands.append('unk_brand')
        else:
            filled_brands.append(row[0])

    df['brand_name'] = filled_brands
    return df


def get_len_feature(col_series, scaler_text_len=None):
    """
    Description:
    This funciton will calculate the word count of the text and standardize it.

    Input: Series, fitted scaler[optional; used during inference]
    Output: standardized text length for each product and object of the fitted scaler
    """
    text_len = col_series.apply(lambda x: len(x.split()))
    if scaler_text_len == None:
        scaler_text_len = StandardScaler()
        scaler_text_len.fit(text_len.values.reshape(-1, 1))
    text_len = scaler_text_len.transform(text_len.values.reshape(-1, 1))
    return text_len, scaler_text_len


def split_text(text):
    if text == 'unk_cat':
        return ["No Label", "No Label", "No Label"]
    return text.split("/")


def split_categories(df):
    """
    Desription:
    This function separates the categories into its three parts.
    Main category, Sub-category 1 and Sub-category 2
    Then it will remove the original category_name field.

    Input: Dataframe having category_name field
    Output: Dataframe with splitted categories
    """
    df['general_cat'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_text(x)))
    df = df.drop('category_name', axis=1)
    return df


def get_is_expensive_feature(df):
    """
    Description:
    This funciton will generate a feature which will tell if the brand is expensive or not.

    Input: Dataframe
    Output: Sparse is_expensive data
    """

    is_expensive_binary = df['brand_name'].apply(lambda x: 1 if x in expensive_brands else 0)
    sparse_shipping = scipy.sparse.csr_matrix(is_expensive_binary.values)
    sparse_shipping = sparse_shipping.reshape(-1, 1)  # Now the shape will be (1111901, 1)
    return sparse_shipping


def vectorize_data(col_data, vectorizer=None):
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
        vectorizer.fit(col_data)
    ohe_data = vectorizer.transform(col_data)
    return ohe_data, vectorizer


def get_shipping_feature(df):
    sparse_shipping = scipy.sparse.csr_matrix(df['shipping'].values)
    sparse_shipping = sparse_shipping.reshape(-1, 1)  # Now the shape will be (1111901, 1)
    return sparse_shipping
