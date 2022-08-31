# Utils to process audio files before feeding to the model #

from posixpath import split

def get_audio_chunks_jl(signal, frame_size, sampling_rate):
    """
    Returns a list of audio chunks from a signal. The chunks are of length
    specified by the frame_size parameter, and can be trimmed for matching annotations of
    the JL-Corpus.
    """
    # Chunk size = Sampling rate x frame size
    chunk_size = int(sampling_rate*frame_size*.001)
    split_file = []
    for i in range(0, len(signal[0][0]), chunk_size):
        split_file.append(signal[0][0][i:chunk_size+i])

    split_file = split_file[1::]
    return split_file

def get_audio_chunks_recola(signal, frame_size, sampling_rate):
    """
    Returns a list of audio chunks from a signal. The chunks are of length
    specified by the frame_size parameter.
    """
    # Chunk size = Sampling rate x frame size
    chunk_size = int(sampling_rate*frame_size*.001)
    split_file = []
    for i in range(0, len(signal[0][0]), chunk_size):
        split_file.append(signal[0][0][i:chunk_size+i])
    return split_file

def get_audio_chunks_semaine(signal, frame_size, sampling_rate):
    """
    Returns a list of audio chunks from a signal. The chunks are of length
    specified by the frame_size parameter
    """
    # Chunk size = Sampling rate x frame size
    chunk_size = int(sampling_rate*frame_size*.001)
    split_file = []
    for i in range(0, len(signal[0][0]), chunk_size):
        split_file.append(signal[0][0][i:chunk_size+i])

    if len(split_file[-1]) < 640:
        split_file = split_file[0:-1]   
    return split_file
