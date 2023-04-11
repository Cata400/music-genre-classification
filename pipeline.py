from utils import *

def test_pipeline(file, window_type, overlap, 
                  window_length, rec_mode):
    '''
    Python script to test the preprocess pipeline
    
    The script prints the MSE and generates 2 graphs:
        1. The original signal and the reconstructed one
        2. Mean Squared Error in each point
    
    Arguments:
        file [strig], path and name of file to be read
        		!MUST BE .wav!
        window_type [string], usually: 'blackman', 'hann' or 'rect'
                              check utils.sigwin() for more details
        overlap [int], the percentage of overlapping between windows
        window_length [int], length of the window in samples
        rec_mode [string], reconstruction mode, usually 'OLA' or 'MEAN'
        		   check utils.sigrec() for more details
    '''
    
    sample_rate, song = wavfile.read(file)
    song = normalization(convert_to_sg_ch(song))
    v_wind = sigwin(song, window_length, window_type, overlap)
    fft = FFT(v_wind, next_pow_of_2(window_length))
    ifft = IFFT(fft, window_length)
    rec = sigrec(ifft.real, overlap, rec_mode)
    
    mod = 'Overlap-add' if rec_mode == 'OLA' else 'Overlap-mean' 
    text = mod + ' method is used at reconstruction\n'
    time = np.linspace(0, np.shape(rec)[0]//sample_rate, np.shape(rec)[0])
    
    plt.plot(time, song[0:np.shape(rec)[0]], 'blue')
    plt.plot(time, rec, 'red')
    plt.xlabel('time [seconds]')
    plt.ylabel('Amplitude (normed)')
    plt.title('Original song vs Reconstructed song\n'+'For window type: ' +
              str(window_type)+', overlap: '+str(overlap) + '%\n and window length: ' + 
              format(1000*window_length/sample_rate,'.2f') + ' ms\n' + text)
    plt.legend(('Original','Reconstructed'))
    plt.show()
    
    mse, mse_points = MSE(song, rec)
    plt.plot(time, mse_points)
    plt.xlabel('time [seconds]')
    plt.ylabel('MSE value')
    plt.title('Mean squared error\n'+'For window type: ' + str(window_type) +
              ', overlap: '+str(overlap) + '%\n and window length: ' + 
              format(1000*window_length/sample_rate,'.2f') + ' ms\n' + text)
    plt.show()
    
    print('MSE value is: ', mse)