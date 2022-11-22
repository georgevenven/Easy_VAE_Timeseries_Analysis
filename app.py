from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from vae import * 
from pre_processing import *

app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

def test_data_injection():
    session['dir'] = '/home/george/Documents/other/test_shortned_data/'
    session['input_type'] = 'raw_audio_files'
    session['remove_empty_space'] = True
    session['empty_window_size'] = 7
    session['empty_threshold'] = 8
    session['segment_length'] = 11 # around 50ms, figure out how to calculate this automatically 
    session['segment_overlap'] = .07 # 7 precent overlap  
    session['segment_size'] = 128
    session['latent_dims'] = 32
    session['batch_size'] = 32
    session['epochs'] = 100
    session['train_valid_split'] = .8
    session['save_epoch_interval'] = 10
    session['root_dir'] = os.getcwd() + '/'

def init():
    root_dir = os.getcwd() + '/'
    session['root_dir'] = root_dir
    session['dir'] = None
    session['input_type'] = None
    session['remove_empty_space'] = None
    session['empty_window_size'] = None
    session['empty_threshold'] = None
    session['segment_length'] = None
    session['segment_overlap'] = None
    session['segment_size'] = None
    session['latent_dims'] = None
    session['batch_size'] = None
    session['epochs'] = None
    session['train_valid_split'] = None
    session['save_epoch_interval'] = None

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('base.html')

@app.route('/home')
def home():

    # have a reset button that will reset all the cookies 

    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/input', methods=['GET', 'POST'])
def input(): 
    if request.method == 'POST':
        # flask get radio button value 
        radio = request.form['filetype']
        dir = request.form['data_location']
        # get data from a bootstrap text field
        print(dir)

        if radio and dir is not None:
            return render_template('adjust_parameters.html')

    return render_template('input.html')

@app.route('/adjust_parameters', methods=['GET', 'POST'])
def adjust_parameters():
    if request.method == 'POST':
        # get data from a bootstrap text field

        # flask, get checkbox value from a form
        remove_empty_space = request.form.get('remove_empty_space')
        empty_window_size = request.form['window_size']
        empty_threshold = request.form['threshold']
        segment_length = request.form['segment_length']
        segment_overlap = request.form['segment_overlap']
        segment_size = request.form['segment_size']
        latent_dims = request.form['latent_dims']
        batch_size = request.form['batch_size']
        epochs = request.form['epoch_num']
        train_valid_split = request.form['train_valid_split']

        # set cookies
        session['remove_empty_space'] = remove_empty_space
        session['empty_window_size'] = empty_window_size
        session['empty_threshold'] = empty_threshold
        session['segment_length'] = segment_length
        session['segment_overlap'] = segment_overlap
        session['segment_size'] = segment_size
        session['latent_dims'] = latent_dims
        session['batch_size'] = batch_size
        session['epochs'] = epochs
        session['train_valid_split'] = train_valid_split

        return render_template('train.html')

    return render_template('adjust_parameters.html')

@app.route('/pre_processing', methods=['GET', 'POST'])
def pre_processing():
    return render_template('pre_processing.html') 


@app.route('/train', methods=['GET', 'POST'])
def train():
    session['epochs'] = 100
    if request.method == 'POST':
        # get the value of the button 
        # button = request.form['train_button']

        # print(button)
        test_data_injection()

        vae = VAE()

        return render_template('train.html', epoch_num=session['epochs'])

    return render_template('train.html', epoch_num=session['epochs'])