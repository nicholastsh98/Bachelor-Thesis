# Create lists to store datagrams and their indexes
import struct
import tkinter as tk
from tkinter import filedialog
from classes import EB200header1
from classes import IFPanheader
from statistics import mean
import datetime
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
from matplotlib.widgets import Slider,CheckButtons


from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
data=[]
data2=[]
data3=[]
data4=[]
timestampdata=[]
import random
import matplotlib.pyplot as plt
def convert_unix_epoch(value):
    # Create a datetime object from the timestamp in seconds
    timestamp_seconds = value // 10 ** 9
    timestamp_datetime = datetime.datetime.utcfromtimestamp(timestamp_seconds)

    # Format the datetime object as a human-readable string
    human_readable_format = timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S')

    return human_readable_format
def IFPANoptionalheaderprint():
    global index
    global IFPanoptionalheadinstance, total_datagrams
    if (tag_value == 10501):  # parsing IFPans optional header
        next_104_bytes = binary_data[ index + 18 + 22 + 36:index + 18 + 22 + 36 + 104+ 4]  # reads next 26 bytes of data after attributes

        unpacked_IFPanoptionalheader_attributes = struct.unpack('<I I h h I I i I I Q I I h h Q h h h h H h q q q I I H H', next_104_bytes) #little endian once optional header is reached
        IFPanoptionalheadinstance = IFPanheader(
            unpacked_IFPanoptionalheader_attributes[0], #freq_low
            unpacked_IFPanoptionalheader_attributes[1], #freqspan
            unpacked_IFPanoptionalheader_attributes[2], #avgtime
            unpacked_IFPanoptionalheader_attributes[3], #avgtype
            unpacked_IFPanoptionalheader_attributes[4], #measuretime
            unpacked_IFPanoptionalheader_attributes[5], #freq_high
            unpacked_IFPanoptionalheader_attributes[6], #demodfreqchannel
            unpacked_IFPanoptionalheader_attributes[7], #demodfreqlow
            unpacked_IFPanoptionalheader_attributes[8], #demodfreqhigh
            unpacked_IFPanoptionalheader_attributes[9], #outputtimestamp
            unpacked_IFPanoptionalheader_attributes[10], #stepfreqnumerator
            unpacked_IFPanoptionalheader_attributes[11], #stepfreqdenom
            unpacked_IFPanoptionalheader_attributes[12], #signalsource
            unpacked_IFPanoptionalheader_attributes[13], #measuremode
            unpacked_IFPanoptionalheader_attributes[14], #measuretimestamp
            unpacked_IFPanoptionalheader_attributes[15], #selectivity
            unpacked_IFPanoptionalheader_attributes[16], #avgtype2
            unpacked_IFPanoptionalheader_attributes[17], #avgtype3
            unpacked_IFPanoptionalheader_attributes[18], #avgtype4
            unpacked_IFPanoptionalheader_attributes[19], #spmenabled
            unpacked_IFPanoptionalheader_attributes[20], #gateenabled
            unpacked_IFPanoptionalheader_attributes[21], #interval
            unpacked_IFPanoptionalheader_attributes[22], #gateoffset
            unpacked_IFPanoptionalheader_attributes[23], #gatelength
            unpacked_IFPanoptionalheader_attributes[24], #fEdge
            unpacked_IFPanoptionalheader_attributes[25], #traceID
        )
        return IFPanoptionalheadinstance
    return None
# def update_plot(new_x, new_y):
#     x.append(new_x)
#     updated.append(new_y)
#     line.set_data(x, updated)
#     ax.relim()
#     ax.autoscale_view()
#     fig.canvas.flush_events()
#main
IFPANLIST = []
root = tk.Tk()
root.withdraw()  # Hide the main window
    # Prompt the user to select a file
file_path = filedialog.askopenfilename(initialdir='C:/Users/nicho/Downloads/Compressed/OneDrive_2023-12-30/BT project backup/recordings',
                                           title='Select a Project File',)
if file_path:
        # Open the selected file
    with open(file_path, 'rb') as file:
        binary_data = file.read()

# Close the GUI window
    root.destroy()
# with open('PScanData.bin', 'rb') as file: #read file, change file name to whichever you want to read
#     binary_data = file.read()
    pattern = b'\x00\x0e\xb2\x00'  # Byte pattern "000xEB200" in hexadecimal, seraches it
    pattern_length = len(pattern)  # stores its length in variable pattern length

    index = 0

datagram_list = []
datagram_indexes = []
channel_data = []
channel_data2 =[]
channel_data3 =[]
channel_data4 =[]

while index < len(binary_data):
    index = binary_data.find(pattern, index)
    if index == -1:
        break

    if index + 16 <= len(binary_data):
        extracted_bytes = binary_data[index:index + 16]
        unpacked_data = struct.unpack('>I H H H H I', extracted_bytes)
        magic_number, version_minor, version_major, sequence_number, seq_number_high, data_size = unpacked_data
        next_2_bytes = binary_data[index + 16:index + 16 + 2]
        tagvalue = struct.unpack('>H', next_2_bytes)
        tag_value = tagvalue[0]  # get tag value
        reserved = struct.unpack('>H', binary_data[index+16+2:index +16+2+ 2])[0]
        length = struct.unpack('>I', binary_data[index +16+2+ 2:index +16+2+ 2+ 4])[0]
        reserved2 =struct.unpack('>4I', binary_data[index +16+2+ 2+4:index +16+2+ 2+ 4+16])[0]
        numberoftracevalues = struct.unpack('>I', binary_data[index + 16 + 2 + 2 + 4+16:index + 16 + 2 + 2 + 4+16 + 4])[0]

        user_data = binary_data[index+16+2+ 2+ 4:index+16+2+ 2+ 4 + length]
        channelnumber = struct.unpack('>I', binary_data[index + 16 + 2 + 2 + 4+16 +4:index + 16 + 2 + 2 + 4+16 + 4+4])[0]
        optionalheaderlength = struct.unpack('>I', binary_data[index + 16 + 2 + 2 + 4+16 +4+4:index + 16 + 2 + 2 + 4+16 + 4+4+4])[0]
        selectorflagslow= struct.unpack('>I', binary_data[index + 16 + 2 + 2 + 4+16 +4+4+4:index + 16 + 2 + 2 + 4+16 + 4+4+4+4])[0]
        selectorflagshigh = struct.unpack('>I', binary_data[
                                               index + 16 + 2 + 2 + 4 + 16 + 4 + 4 + 4+4:index + 16 + 2 + 2 + 4 + 16 + 4 + 4 + 4 + 4+4])[
            0]

        datagram = EB200header1(
            magic_number,
            version_minor,
            version_major,
            sequence_number,
            seq_number_high,
            data_size,
            tag_value,
            reserved,
            length,
            reserved2,
            user_data,
            numberoftracevalues,
            channelnumber,
            optionalheaderlength,
            selectorflagslow,
            selectorflagshigh,
        )
        IFPANinstance = IFPANoptionalheaderprint()  # 10501
        if IFPANinstance is not None:
            IFPANLIST.append(IFPANinstance)
        num_indexes = len(IFPANLIST)


        if tag_value == 10501: #if it is IFPAN datagram
            # Store the datagram and its index in the lists
            #manual parsing as certain parameters required
            freq_low = struct.unpack('<I', binary_data[index + 18 + 22 + 36:index + 18+ 22 + 36 + 4])
            freq_span = struct.unpack('<I', binary_data[index + 18 + 22 + 36 + 4:index + 18 + 22 + 36 + 4 + 4])
            measurementtimestamp = struct.unpack('<Q', binary_data[index + 18 + 22 + 36 + 52:index + 18 + 22 + 36 + 60])
            measurementtimestamp = measurementtimestamp[0]
            measurementtimestamp = convert_unix_epoch(measurementtimestamp)
            timestampdata.append(measurementtimestamp)

            datagram_list.append(datagram)
            datagram_indexes.append(index) #number of IFPAN datagrams

    # Update the index to search for the next occurrence of the pattern
            format_string = f'<{numberoftracevalues}h'


            channel_data_size = numberoftracevalues * 2  # Assuming each value is 2 bytes (INT16)

            if index + 184 + channel_data_size <= len(binary_data):
                channel_values = struct.unpack(format_string, binary_data[
                                                          index + 184:index +184+channel_data_size])


                channel_data.append(channel_values)
            else:
                print("Not enough data to unpack channel data.")

            if index + 184 + channel_data_size+channel_data_size <= len(binary_data):
                channel_values1 = struct.unpack(format_string, binary_data[
                                                              index + 184 + channel_data_size : index + 184 + channel_data_size + channel_data_size])

                channel_data2.append(channel_values1)
            else:
                print("Not enough data to unpack channel data.")

            if index + 184 + 2*channel_data_size+channel_data_size <= len(binary_data):
                channel_values = struct.unpack(format_string, binary_data[
                                                              index + 184 +2*channel_data_size:index + 184 +2* channel_data_size + channel_data_size])

                channel_data3.append(channel_values)
            else:
                print("Not enough data to unpack channel data.")

            if index + 184 + 3*channel_data_size+channel_data_size <= len(binary_data):
                channel_values = struct.unpack(format_string, binary_data[
                                                              index + 184 +3*channel_data_size:index + 184 +3*channel_data_size +channel_data_size])

                channel_data4.append(channel_values)
            else:
                print("Not enough data to unpack channel data.")

    # Now, unpack additional data following IFPAN
    # Define the format string for the additional data (adjust as needed)

    index += 1

# Now you can access the stored datagrams by their indexes
for i, index in enumerate(datagram_indexes):
    print(f"Datagram at index {i}: {datagram_list[i]}")
number_of_indexes = len(datagram_list)


#fixed


def format_frequency(value, pos):
    return f'{value / 1e6:} '

if channel_data:
    print("Channel Data:")
    for i, values in enumerate(channel_data):
        print(f"IFPAN  1_ {i + 1}:{values}")
        data.append(values)
# if channel_data:
#     print("Channel Data:")
#     for i, values in enumerate(channel_data2):
#         print(f"IFPAN  2_ {i + 1}:{values}")
#         data2.append(values)
# if channel_data:
#     print("Channel Data:")
#     for i, values in enumerate(channel_data3):
#         print(f"IFPAN  3_ {i + 1}:{values}")
#         data3.append(values)
# if channel_data:
#     print("Channel Data:")
#     for i, values in enumerate(channel_data4):
#         print(f"IFPAN  4_ {i + 1}:{values}")
#         data4.append(values)
def toggle_y_axis(label):
    global fixed_y_axis
    fixed_y_axis = not fixed_y_axis
    update(None)
#modify data
updated_data = []
for index in range(len(data)):
    updated_index = tuple(value / 10 for value in data[index])
    updated_data.append(updated_index)
micro_symbol = '\u00B5'
# print(updated)
freq_low = freq_low[0]
freq_low = freq_low / 10 ** 6
freq_span = freq_span[0]
freq_span = freq_span / 10 ** 6

lower_frequency = freq_low-freq_span/2
step_size = freq_span/numberoftracevalues
x = [(i * step_size) + lower_frequency for i in range(len(updated_data[i]))]


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)  # Adjust to make room for labels and slider

# Initial index
first_timestamp = datetime.datetime.strptime(timestampdata[0], '%Y-%m-%d %H:%M:%S')
last_timestamp = datetime.datetime.strptime(timestampdata[-1], '%Y-%m-%d %H:%M:%S')

# Calculate the time difference
time_difference = last_timestamp - first_timestamp

# Print the total time taken (in seconds)
time=f"time taken: {time_difference.total_seconds()} seconds"
initial_index = 0
micro_symbol = '\u00B5'
text = f"Timestamp: {timestampdata[index]}"
all_data = [y for data in updated_data for y in data]
min_data = int(np.floor(min(all_data)))  # Round down to nearest integer
max_data = int(np.ceil(max(all_data)))
#fixed_y_min = min(all_data)-5
#fixed_y_max = max(all_data)+5
# Function to update the plot

import plotly.graph_objects as go
import plotly.express as px
import datetime
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

# Your existing data processing and calculations

# Create Dash app
# Your existing data processing and calculations

# Convert min and max values to integers
min_data = int(np.floor(min(all_data)))  # Round down to nearest integer
max_data = int(np.ceil(max(all_data)))   # Round up to nearest integer

# Create Dash app
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Your existing data processing and calculations

# Create Dash app

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    dbc.Alert(id='signal-alert', color='info', dismissable=False),
    dcc.Graph(id='plot'),
    html.Div(id='index-display'),
    html.Label('Select Index:'),
    dcc.Slider(
        id='index-slider',
        min=0,
        max=len(updated_data) - 1,
        value=0,
        marks={i: str(i) for i in range(len(updated_data))}
    ),
    html.Label('Set Threshold:'),
    dcc.Slider(
        id='threshold-slider',
        min=min_data,
        max=max_data,
        value=(min_data + max_data) / 2,
        marks={i: str(i) for i in range(min_data, max_data + 1)},
        tooltip={'placement': 'bottom'}
    ),
    dcc.Checklist(
        id='toggle-y-axis',
        options=[{'label': 'Fixed Y-Axis', 'value': 'fixed-y-axis'}],
        value=['fixed-y-axis']
    ),
    html.Div([
        html.H2('Identified Signals Bandwidths'),
        dash_table.DataTable(
            id='bandwidth-table',
            columns=[
                {'name': 'Signal', 'id': 'Signal'},
                {'name': 'Bandwidth', 'id': 'Bandwidth'},
                {'name': 'Start Frequency', 'id': 'Start Frequency'},
                {'name': 'End Frequency', 'id': 'End Frequency'},
                {'name': 'Center Frequency', 'id': 'Center Frequency'}
            ],
            style_table={'overflowX': 'auto'},
        )
    ]),
    html.Div(id='timestamp-output')
])

# Callback to update the plot based on sliders
@app.callback(
    [Output('plot', 'figure'), Output('bandwidth-table', 'data'), Output('index-display', 'children'),Output('signal-alert', 'children')],
    [Input('index-slider', 'value'), Input('threshold-slider', 'value'), Input('toggle-y-axis', 'value')]
)

def update_plot(selected_index, threshold, toggle_value):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=updated_data[selected_index], mode='lines+markers', name='Data'))
    index_display = html.Div(f"Selected Index: {selected_index}", style={'margin-top': '10px', 'font-weight': 'bold'})


    # Draw a red line at the threshold on the y-axis
    fig.add_shape(
        type='line',
        x0=min(x),
        y0=threshold,
        x1=max(x),
        y1=threshold,
        line=dict(color='red', width=2, dash='dash'),
    )

    # Highlight points above the threshold in red
    above_threshold_x = [x[i] for i, y in enumerate(updated_data[selected_index]) if y > threshold]
    above_threshold_y = [y for y in updated_data[selected_index] if y > threshold]
    fig.add_trace(go.Scatter(x=above_threshold_x, y=above_threshold_y, mode='markers', marker=dict(color='red'),
                             name='Above Threshold'))

    # Identify individual signals based on index proximity
    bandwidth_data =[]
    signals = []
    bandwidths = []
    colors = ['blue', 'green', 'orange', 'purple', 'cyan']  # Define colors for signals (add more if needed)
    color_idx = 0

    if len(above_threshold_x) > 0:
        signal = [above_threshold_x[0]]
        for i in range(1, len(above_threshold_x)):
            if above_threshold_x[i] - above_threshold_x[i - 1] <= 0.2:  # Adjust the threshold for signal closeness
                signal.append(above_threshold_x[i])
            else:
                signals.append(signal)
                signal = [above_threshold_x[i]]

        signals.append(signal)

        # Calculate bandwidth for each identified signal and highlight on the plot
        for signal in signals:
            if len(signal) > 1:
                signal_bandwidth = signal[-1] - signal[0]
                bandwidths.append(signal_bandwidth)

                # Highlight signal on the plot
                fig.add_trace(go.Scatter(
                    x=signal,
                    y=[threshold] * len(signal),
                    mode='markers',
                    marker=dict(color=colors[color_idx]),
                    name=f'Signal {color_idx + 1}'
                ))
                color_idx = (color_idx + 1) % len(colors)

    print("Bandwidths of identified signals:", bandwidths)
    for idx, signal in enumerate(signals):
        if len(signal) > 1:
            signal_bandwidth = signal[-1] - signal[0]
            bandwidths.append(signal_bandwidth)

            start_freq = signal[0]  # Start frequency of the signal
            end_freq = signal[-1]  # End frequency of the signal
            mean_freq = sum(signal) / len(signal)  # Mean frequency of the signal


            # Add data for each signal to bandwidth_data
            fig.update_layout(
                title='IFPAN Data ',
                xaxis_title='Frequencies (MHz)',
                yaxis_title=f'Signal Strength (db{micro_symbol})',
                annotations=[
                    dict(
                        text=f"Timestamp: {timestampdata[selected_index]}",
                        x=0.02,
                        y=0.02,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=10, color='red')
                    ),
                    dict(
                        text=f"Total {time}",
                        x=0.02,
                        y=0.07,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=10, color='red')
                    )
                ]
            )

            bandwidth_data.append({
                'Signal': f'Signal {idx + 1}',
                'Bandwidth': f'{signal_bandwidth:.2f} MHz',
                'Start Frequency': f'{start_freq:.2f} MHz',
                'End Frequency': f'{end_freq:.2f} MHz',
                'Center Frequency': f'{mean_freq:.2f} MHz'
            })

    count = len([y for y in updated_data[selected_index] if y > threshold])

    alert_message = dbc.Alert(f'There is/are {idx+1} identified signal/signals above the threshold with {count} data points above the threshold.', color='info',
                              dismissable=True) if count > 0 else None
    # Code to identify signals and calculate bandwidths (as described in previous interactions)

    # bandwidth_data = [{'signal': f'Signal {idx + 1}', 'bandwidth': f'{bandwidth:.2f} MHz'}
    #                   for idx, bandwidth in enumerate(bandwidths)]
    yaxis_settings = {}
    if 'fixed-y-axis' in toggle_value:
        yaxis_settings['fixedrange'] = True
    else:
        yaxis_settings['fixedrange'] = False

    if yaxis_settings.get('fixedrange'):
        yaxis_settings['range'] = [min_data, max_data]  # Define your custom range here

    fig.update_layout(yaxis=yaxis_settings)
    return fig, bandwidth_data, index_display, alert_message

if __name__ == '__main__':
    app.run_server(debug=True)


