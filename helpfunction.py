from copy import deepcopy

import obspy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy as sp
import scipy.signal
from scipy.signal import convolve, correlate

def green_function_2d(time, distance, speed=3):
    """
    2-D Green's function.
    
    :param time: Time range (s)
    :param distance: Source-receiver distance (km)
    :param speed: Medium speed between the source and the receiver (km/s)
    """
    t_arrival = distance / speed
    green_function = np.heaviside(time - t_arrival, 0)
    idx = time > t_arrival
    green_function[idx] /= np.sqrt(time[idx]**2 - t_arrival**2)
    
    return green_function

def green_function_2d_integral(time, distance, speed=3):
    """ 
    Integral of 2-D Green's function.
    
    :param time: Time range (s)
    :param distance: Source-receiver distance (km)
    :param speed: Medium speed between the source and the receiver (km/s)
    """
    t_arrival = distance / speed
    green_function_integral = np.heaviside(time - t_arrival, 0)
    idx = time > t_arrival
    green_function_integral[idx] = np.log((time[idx] + np.sqrt(time[idx]**2 - t_arrival**2)) / t_arrival)
    
    return green_function_integral

def ricker(t, t0=0, fc=.1, **kwargs):
    """
    Ricker wavelet function.
    
    :param t: Time range (s)
    :param t0: Initial time (s)
    :fc: Frequency (Hz)
    """
    #tc = t - t0
    #tmp = (np.pi * fc * tc)**2
    #return (1 - 2*tmp) * np.exp(-tmp)
    #below 0.2Hz
    tc = t - t0
    sigma = 2
    result = 2*(1-(tc/sigma)**2)*np.exp(-tc**2/(2*sigma**2))/(np.sqrt(3*sigma)*np.power(np.pi, 0.25))
    nor = result/np.max(np.abs(result))
    return nor

def normalize(a, axis=1):
    """
    Normalize function.
    
    :param a: A sequence that requires normalization 
    :param axis: Dimensionality（axis=0:Find the maximum value for each column，axis=1:Find the maximum value for each row）
    """
    if a.ndim == 1:
        return a / np.abs(a).max()
    else:
        return a / np.abs(a).max(axis=axis).reshape(-1, 1)

def greenfunction_between_source_receiver(time, source_radius, source_azimuth, receiver_location,
        source_time_function, gf_func=green_function_2d, c=3):
    """
    Calculate Green's function between source and receiver.
    
    :param time: Time (s)
    :param source_radius: Radius of the circle which distributed sources (km)
    :param source_azimuth: The azimuth of source distribution (degree)
    :param receiver_location: The location the receiver (km)
    :param source_time_function: Source time function
    :param gf_func: The type of Green's function
    :param c: Medium speed (km/s)
    """
    if np.asarray(c).size == 1:
        c = np.full(source_azimuth.shape, c)

    d =  np.sqrt(
        (source_radius*np.cos(source_azimuth) - receiver_location)**2
        + (source_radius*np.sin(source_azimuth) - 0)**2
    )
    gf = []
    for _stf, _d, _c in zip(source_time_function, d, c):
        greenfunction_between_source_receiver = gf_func(time, _d, _c)
        gf.append(greenfunction_between_source_receiver)

    return np.vstack(gf)

def source_convolve_with_greenfunction(source_time_function, green_function):
    """
    Calculate source convolve with green's function.
    
    :param source_time_function: Source time function
    :param green_function: Green's function
    """
    u = []
    for stf, gf in zip(source_time_function, green_function):
        u.append(normalize(convolve(stf, gf, mode='full')[:gf.size]))
        
    return np.vstack(u)

def get_displacement_from_sources(time, source_amplitude,
                 source_time,
                 gf_func=green_function_2d, source_func=ricker,
                 source_radius=200, source_azimuth=np.arange(0, 2*np.pi, np.pi/360),
                 receiver_location_1=-100, receiver_location_2=100, c=3, **kwargs):
    """
    Calculate displacement given source and structure.
    
    :param time: Time (s)
    :param source_amplitude: Source amplitude
    :param source_time: Source excitation time (s)
    :param gf_func: The type of Green's function
    :param source_func: The type of source
    :param source_radius: Radius of the circle which distributed sources (km)
    :param source_azimuth: The azimuth of source distribution (degree)
    :param receiver_location_1: The location of the first receiver (km)
    :param receiver_location_2: The location of the second receiver (km)
    """
    source_amplitude = np.asarray(source_amplitude)
    source_azimuth = np.asarray(source_azimuth)
    c = np.asarray(c)
    source_time = np.asarray(source_time)
    stf = source_func(time, source_time.reshape(-1, 1))
    #tmax = stf.shape[1]
    #noise = np.random.normal(loc=0, scale=0.1, size=(1,tmax))
    #stf = stf+noise
    '''
    plt.plot(noise[source_time-300:source_time+300], 'b', label='noise')
    plt.plot(stf[source_time-300:source_time+300], 'r', label='stf')
    plt.savefig('stf.png')
    plt.close()
    '''
    gf1 = greenfunction_between_source_receiver(time, source_radius, source_azimuth, receiver_location_1,
                 source_time_function=stf, gf_func=gf_func, c=c)
    u1 = source_amplitude.reshape(-1, 1) * source_convolve_with_greenfunction(stf, gf1)
    gf2 = greenfunction_between_source_receiver(time, source_radius, source_azimuth, receiver_location_2,
                 source_time_function=stf, gf_func=gf_func, c=c)
    u2 = source_amplitude.reshape(-1, 1) * source_convolve_with_greenfunction(stf, gf2)
    
    return np.sum(u1, axis=0), np.sum(u2, axis=0)
#    return stf, gf1, u1, gf2, u2

def ground_truth(time,source_func=ricker,
                        receiver_location_1=-100, receiver_location_2=100, c_receiver=3, shift=20, delta=.1, **kwargs):
    """
    Calculate true displacement use Green's function.
    
    :param source_func: The type of source
    :param receiver_location_1: The location of the first receiver (km)
    :param receiver_location_2: The location of the second receiver (km)
    :param c_receiver: Medium speed between two receivers (km/s)
    :param shift: Interval
    """
    stf = source_func(time, shift)
    ishift = int(shift / delta)
    stf = correlate(stf, stf, 'full')[-time.size-ishift:]
    x = abs(receiver_location_1 - receiver_location_2)
    gf = green_function_2d_integral(time, x, c_receiver)
    u12 = convolve(stf, gf, mode='full')[ishift : gf.size+ishift-1]
    u = np.concatenate([-u12[::-1], [0], -u12])
    # TODO: check artifact
    u *= np.hanning(u.size)

    return u

def cross_correlation(u1, u2):
    """
    Cross correlation function.
    
    :param u1: The first sequence
    :param u2: The second sequence
    """
    xc = []
    for _u1, _u2 in zip(u1, u2):
        xc.append(correlate(_u1, _u2, 'full'))

    return np.vstack(xc)

def time_xc(t):
    """
    Function to extend an arithmetic sequence using the original one.
    
    :param t: Original time sequence
    """
    return np.linspace(-t[-1], t[-1], 2*t.size-1)

def running_mean(a, n):
    """
    Running mean function.
    
    :param a: Sequence need running mean
    :param n: Size of the window
    """
    return convolve(a, np.ones(n)/n, mode='same')

def plt_input(source_amplitude,
                 source_radius=200, source_azimuth=np.arange(0, 2*np.pi, np.pi/360),
                 receiver_location_1=-100, receiver_location_2=100, step=None, c=3, **kwargs):
    """
    Plot input model parameters function.
    
    :param source_amplitude: Source amplitude
    :param source_radius: Radius of the circle which distributed sources (km)
    :param source_azimuth: The azimuth of source distribution (degree)
    :param receiver_location_1: The location of the first receiver (km)
    :param receiver_location_2: The location of the second receiver (km)
    :param step: Drawing step
    """
    if step is None:
        step = int(np.ceil(source_azimuth.size / 100))

    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], projection='polar')
    ax3 = fig.add_subplot(gs[2], projection='polar')
    
    # Schematic of geometry
    ax1.axhline(0, c='k')
    ax1.axvline(0, c='k')
    for r, txt in zip([receiver_location_1, receiver_location_2],
                      [r'$r_1$', r'$r_2$']):
        ax1.scatter(r, 0, s=200, marker='^', c='b')
        ax1.text(r, -60, txt, fontsize=20)
    theta = source_azimuth[::step]
    ax1.scatter(source_radius*np.cos(theta), source_radius*np.sin(theta), s=50, marker='*', c='r')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    xlim = 1.1 * max(abs(receiver_location_1), source_radius)
    ax1.set_xlim(-xlim, xlim)
    ax1.set_ylim(-xlim, xlim)
    ax1.set_aspect(1)
    
    # Source strength
    ax2.fill_between(source_azimuth, 0, source_amplitude, alpha=.5, color='tab:blue')
    ax2.set_title('Source amplitude', y=1.1)
    
    # Speed
    ax3.fill_between(source_azimuth, 0, c, alpha=.5, color='tab:orange')
    ax3.set_title('Speed (km/s)', y=1.1)
    
#     for ax in axes:
#         ax.set_theta_zero_location('N')
#         ax.set_theta_direction(-1)

    return

def plt_waveform(u1, u2, xc, direct, source_azimuth, time,
                 xlim=300, step=10, **kwargs):
    """
    Plot waveform function.
    
    :param u1: First receiver's data
    :param u1: Second receiver's data
    :param xc: The cross correlation of u1 and u2
    :param source_azimuth: The azimuth of source distribution (degree)
    :param time: Time (s)
    :param xlim: The limit of x axis (s)
    :param step: Drawing step
    """
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[.7, .3])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    
    for ax, u, t in zip(
        [ax1, ax2, ax3],
        [u1, u2, xc],
        [time, time, time_xc(time)],
    ):
        vmax = np.abs(u).max()
        ax.pcolormesh(t[::step], np.rad2deg(source_azimuth)[::step], u[::step, ::step],
                      cmap='seismic', vmin=-vmax, vmax=vmax)
        

    xc_stack = xc.sum(axis=0)
    dt_xc = np.diff(xc_stack)
    mid = int(dt_xc.size / 2)
    dt_xc = np.concatenate([dt_xc[:mid], [0], dt_xc[mid:]])
    for ax in [ax4, ax5]:
        ax.plot(time_xc(time), normalize(direct), c='k', ls='--', label='True')
        ax.plot(time_xc(time), normalize(xc_stack), c='r', alpha=.5, label='Correlation stack')
    ax4.legend()
    ax4.set_ylabel('Norm. Amplitude')

    for ax in [ax3, ax4, ax5]:
        ax.set_xlim(-xlim/2, xlim/2)
        ax.axvline(0, c='k')
    for ax in [ax1, ax2]:
        ax.set_xlim(0, xlim)
#     for ax in axes[1, :2]:
#         ax.axis('off')
    ax1.set_ylabel(r'Source azimuth ($^\circ)$')
    ax1.set_yticks(np.linspace(0, 360, 5))
    for ax, t in zip(
        [ax1, ax2, ax3],
        [r'$u_1$', r'$u_2$', r'$u_1 \star u_2$'],
    ):
        ax.text(.1, .9, t, fontsize=20, transform=ax.transAxes)
    for ax in [ax4, ax5]:
        ax.set_xlabel('Time (s)')
    for ax in [ax2, ax3, ax5]:
        ax.set_yticklabels([])
    ax3.set_xticklabels([])
        
    return

def plt_processing(raw, processed, threshold, time, **kwargs):
    """
    Function to plot the data after pre-processing.
    
    :param raw: The raw data without pre-processing
    :param processed: The processed data after pre-processing
    :param threshold: The threshold in one-bit
    :param time: Time (s)
    """
    fig, ax = plt.subplots()
    ax.plot(time, raw, ls='--', label='Raw', c='k')
    ax.plot(time, processed, alpha=.5, c='r', label='Processed')
    ax.fill_between(time, -threshold, threshold, alpha=.3, color='gray', label='Threshold')
    ax.set_xlim(0, 300)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized amplitude')
    ax.legend()
    
    return

def plot_src(src_radius, src_theta, sta1_loc, sta2_loc, c=3):
    '''
    plot the distribution of noise source and station
    '''
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])

    #Schematic of geometry
    ax1.axhline(0, c='k')
    ax1.axvline(0, c='k')
    for r,txt in zip([sta1_loc, sta2_loc],[r'$r_1$', r'$r_2$']):
        ax1.scatter(r, 0, s=200, marker='^',c='b')
        ax1.text(r, -60, txt, fontsize=20)
    for i in range(0, len(src_radius)):
        ax1.scatter(src_radius[i]*np.cos(src_theta[i]), src_radius[i]*np.sin(src_theta[i]), s=5, marker='*',c='r')
    ax1.set_xlabel('X(km)')
    ax1.set_ylabel('Y(km)')
    xlim = 1.1*max(abs(sta1_loc), max(src_radius))
    ax1.set_xlim(-xlim, xlim)
    ax1.set_ylim(-xlim, xlim)
    ax1.set_aspect(1)

    plt.savefig('source_distribution.png')
    plt.close()
    return

def whiten(tr, freqmin, freqmax):
    nsamp = tr.stats.sampling_rate
    n = len(tr.data)
    if n == 1:
        return tr
    else:
        frange = float(freqmax) - float(freqmin)
        nsmo = int(np.fix(min(0.01, 0.5 * (frange))* float(n)/nsamp))
        f = np.arange(n) * nsamp /(n -1.)
        JJ = ((f > float(freqmin)) & (f < float(freqmax))).nonzero()[0]

        # 信号的傅里叶变换
        FFTs = np.fft.fft(tr.data)
        FFTsW = np.zeros(n) + 1j * np.zeros(n)

        # 
        smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo+1))**2)
        FFTsW[JJ[0]:JJ[0]+nsmo+1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0]+nsmo+1]))

        FFTsW[JJ[0]+nsmo+1:JJ[-1]-nsmo] = np.ones(len(JJ) - 2 * (nsmo+1))\
        * np.exp(1j * np.angle(FFTs[JJ[0]+nsmo+1:JJ[-1]-nsmo]))

        smo2 = (np.cos(np.linspace(0., np.pi/2., nsmo+1))**2.)
        espo = np.exp(1j * np.angle(FFTs[JJ[-1]-nsmo:JJ[-1]+1]))
        FFTsW[JJ[-1]-nsmo:JJ[-1]+1] = smo2 * espo

        whitedata = 2. * np.fft.ifft(FFTsW).real
        
        tr.data = np.require(whitedata, dtype="float32")

        return tr


def cor_normalize(tr, clip_factor=6, clip_weight=10, norm_win=10, norm_method="lbit"):
    if norm_method == 'clipping':
        lim = clip_factor * np.std(tr.data)
        tr.data[tr.data > lim] = lim
        tr.data[tr.data < -lim] = -lim
    elif norm_method == "clipping_iter":
        lim = clip_factor * np.std(np.abs(tr.data))

        while tr.data[np.abs(tr.data) > lim] != []:
            tr.data[tr.data > lim] /= clip_weight
            tr.data[tr.data < -lim] /= clip_weight
    elif norm_method == "ramn":
        lwin = int(tr.stats.sampling_rate * norm_win)
        st = 0
        N = lwin
        while N < tr.stats.npts:
            win = tr.data[st:N]
            w = np.mean(np.abs(win)) / (2. * lwin + 1)
            tr.data[st + lwin //2] /= w
            st += 1
            N += 1
          
        # 波形最前端以及最后端振幅逐渐衰减到 0，类似于上面
        taper = get_window(tr.stats.npts)
        tr.data *= taper
        
    elif norm_method == "lbit":
        tr.data = np.sign(tr.data)
        tr.data = np.float32(tr.data)
    return tr

def get_window(N, alpha=0.2):
    window = np.ones(N)
    x = np.linspace(-1., 1., N)
    ind1 = (abs(x) > 1 - alpha) * (x < 0)
    ind2 = (abs(x) > 1 - alpha) * (x > 0)
    window[ind1] = 0.5 * (1- np.cos(np.pi *(x[ind1]+1)/alpha))
    window[ind2] = 0.5 * (1- np.cos(np.pi *(x[ind2]-1)/alpha))
    # print(window)
    return window

def plotTrace(tr, starttime = 0, endtime= 30):
    plt.figure(figsize=(10, 4))
    #plt.figure(figsize=(7, 3))
    depmax = np.max(tr.data)
    depmin = np.min(tr.data)
    maxY = np.fabs(depmin)
    if depmax > maxY:
        maxY = depmax
    myTimes = tr.times()
    myData = tr.data / depmax * 0.9 
    plt.plot(myTimes, myData, color='gray', linewidth=1)
    plt.xlim(starttime, endtime)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def correlateNoise(st, stations):
    st1 = st.select(station=stations[0]).sort()
    st2 = st.select(station=stations[1]).sort()
    for i in st1:
        try:
            data1 = np.vstack((data1, i.data))
        except:
            data1 = i.data
    for i in st2:
        try:
            data2 = np.vstack((data2, i.data))
        except:
            data2 = i.data
    if len(data1)==len(data2):
        for i in range(0, len(data1)-1):
            xcorr = np.correlate(data1[i], data2[i], 'same')
            try: 
                # build array with all correlations
                corr = np.vstack((corr, xcorr))
            except: 
                # if corr doesn't exist yet
                corr = xcorr
        # stack the correlations; normalize
        stack = np.sum(corr, 0)
        stack = stack / float((np.abs(stack).max()))    
        print ("...done")
    else:
        print('wrong')
    return corr, stack

def segmentation(stream, interval):
    st = stream[0].stats.starttime
    et = stream[0].stats.endtime
    timewin = st+interval
    new_stream1 = obspy.core.stream.Stream()
    #new_stream2 = obspy.core.stream.Stream()
    while timewin < et:
        tmp1 = stream[0].slice(timewin-interval, timewin)
        new_stream1.append(tmp1)
        tmp2 = stream[1].slice(timewin-interval, timewin)
        new_stream1.append(tmp2)
        timewin += interval
    return new_stream1

def plot_record(name1, name2, l):
    u1 = np.load(name1)
    u2 = np.load(name2)

    length = l
    u1 = u1[0:length]
    u2 = u2[0:length]
    stats = obspy.core.trace.Stats()
    stats['station'] = 'st1'
    stats.network = 'TES'
    stats.sampling_rate = 1.0
    stats.npts = length
    stats.starttime = obspy.UTCDateTime(2009, 1, 1, 12, 0, 0)
    seis1 = obspy.core.trace.Trace(data = u1, header=stats)
    seis1.write('seis1.sac', format='sac')
    stats['station'] = 'st2'
    seis2 = obspy.core.trace.Trace(data = u2, header=stats)
    seis2.write('seis2.sac', format='sac')
    return seis1, seis2

    '''
    u1 = np.sign(u1)
    u2 = np.sign(u2)
    u1 = np.float32(u1)
    u2 = np.float32(u2)
    u1 = whiten(u1, 0.02, 0.07)
    u2 = whiten(u2, 0.02, 0.07)

    xcorr = np.correlate(u1, u2, 'same')

    fig = plt.figure(figsize=(8,4))
    gs = GridSpec(3, 1, figure = fig)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(u1)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(u2)

    ax3 = fig.add_subplot(gs[2])
    n = int(len(xcorr)/2)
    print(xcorr[n-300:n+300].shape)
    ax3.plot(np.arange(-300,300), xcorr[n-300:n+300])
    ax3.set_xlim([-300, 300])
    
    #plotTrace(seis1.filter, starttime = 0, endtime= 3600)
    #plotTrace(seis2, starttime = 0, endtime= 3600)
    st = obspy.core.stream.Stream(traces=[seis1, seis2])
    st.detrend('linear')

    st.taper(max_percentage=0.05, type='cosine')
    
    freq_range = [1/16, 1/6]

    st.filter('bandpass', freqmin=freq_range[0], freqmax=freq_range[1], corners=4, zerophase=True)
    #st.plot()
    stp = segmentation(st, 999)
    for i in stp:
        i = cor_normalize(i, norm_win=10,norm_method="lbit")
    for i in stp:
        i = whiten(i, freq_range[0], freq_range[1])
    stack, xcorr = correlateNoise(stp, ['st1', 'st2'])
    print(stack.shape)
    np.save('stack.npy', stack)
    fig = plt.figure(figsize=(8,4))
    gs = GridSpec(1, 1, figure = fig)
    ax3 = fig.add_subplot(gs[0])
    n = int(len(xcorr)/2)
    ax3.plot(np.arange(-200,200), xcorr[n-200:n+200], 'k')
    ax3.set_xlim([-200, 200])
    ax3.set_ylim([-1.2, 2])
    plt.savefig('./corrle.png')
    plt.close()

    return
    '''