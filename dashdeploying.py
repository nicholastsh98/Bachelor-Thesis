# Create lists to store datagrams and their indexes
import struct
import tkinter as tk
from tkinter import filedialog

from classes import IFPanheader
from classes import DFPANheader
import datetime
import dash_bootstrap_components as dbc
data=[]
data2=[]
data3=[]
data4=[]
timestampdata=[]
import random
import matplotlib.pyplot as plt
def DFPanoptionalheaderprint():
    global DFPanheaderinstance, total_datagrams
    if (attributes == 1401):  # parsing PScams optional header
        next_128_bytes = binary_data[
                         index + 16 + 4 + 8:index + 16 + 4 + 8 + 128]  # reads next 26 bytes of data after advabced attributes
        unpacked_DFPanoptionalheader_attributes = struct.unpack('<I I I i i I I I i H h i i I I Q h h h h f h h f f I I Q h h h h h 6B Q',next_128_bytes)
        DFPanheaderinstance = DFPANheader(
            unpacked_DFPanoptionalheader_attributes[0],
            unpacked_DFPanoptionalheader_attributes[1],
            unpacked_DFPanoptionalheader_attributes[2],
            unpacked_DFPanoptionalheader_attributes[3],
            unpacked_DFPanoptionalheader_attributes[4],
            unpacked_DFPanoptionalheader_attributes[5],
            unpacked_DFPanoptionalheader_attributes[6],
            unpacked_DFPanoptionalheader_attributes[7],
            unpacked_DFPanoptionalheader_attributes[8],
            unpacked_DFPanoptionalheader_attributes[9],
            unpacked_DFPanoptionalheader_attributes[10],
            unpacked_DFPanoptionalheader_attributes[11],
            unpacked_DFPanoptionalheader_attributes[12],
            unpacked_DFPanoptionalheader_attributes[13],
            unpacked_DFPanoptionalheader_attributes[14],
            unpacked_DFPanoptionalheader_attributes[15],
            unpacked_DFPanoptionalheader_attributes[16],
            unpacked_DFPanoptionalheader_attributes[17],
            unpacked_DFPanoptionalheader_attributes[18],
            unpacked_DFPanoptionalheader_attributes[19],
            unpacked_DFPanoptionalheader_attributes[20],
            unpacked_DFPanoptionalheader_attributes[21],
            unpacked_DFPanoptionalheader_attributes[22],
            unpacked_DFPanoptionalheader_attributes[23],
            unpacked_DFPanoptionalheader_attributes[24],
            unpacked_DFPanoptionalheader_attributes[25],
            unpacked_DFPanoptionalheader_attributes[26],
            unpacked_DFPanoptionalheader_attributes[27],
            unpacked_DFPanoptionalheader_attributes[28],
            unpacked_DFPanoptionalheader_attributes[29],
            unpacked_DFPanoptionalheader_attributes[30],
            unpacked_DFPanoptionalheader_attributes[31],
            unpacked_DFPanoptionalheader_attributes[32],
            unpacked_DFPanoptionalheader_attributes[33],
            unpacked_DFPanoptionalheader_attributes[34],
        )
        return DFPanheaderinstance
    return None
def convert_unix_epoch(value):
    # Create a datetime object from the timestamp in seconds
    timestamp_seconds = value // 10 ** 9
    timestamp_datetime = datetime.datetime.utcfromtimestamp(timestamp_seconds)
    # Format the datetime object as a human-readable string
    human_readable_format = timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return human_readable_format
class EB200DatagramFormat:
    def __init__(self, magic_number, version_major, version_minor, sequence_number, seq_number_high, data_size, attributes,
                 length, number_of_trace_values, channel_number, optional_header_length, selector_flags):
        self.magic_number = magic_number
        self.version_major = version_major
        self.version_minor = version_minor
        self.sequence_number = sequence_number
        self.seq_number_high = seq_number_high
        self.data_size = data_size
        self.attributes = attributes
        self.length = length
        self.number_of_trace_values = number_of_trace_values
        self.channel_number = channel_number
        self.optional_header_length = optional_header_length
        self.selector_flags = selector_flags

def EB200headerprint():
    print(f"Magic Number: {hex(datagram.magic_number)}")
    print(f"Version Major: {hex(datagram.version_major)}")
    print(f"Version Minor: {hex(datagram.version_minor)}")
    print(f"Sequence Number: {hex(datagram.sequence_number)}")
    print(f"Sequence Number High: {hex(datagram.seq_number_high)}")
    print(f"Attributes: {datagram.attributes}")
    print(f"Data Size: {datagram.data_size}")
    print(f"Length: {datagram.length}")
    print(f"Number of Trace Values: {datagram.number_of_trace_values}")
    print(f"Channel Number: {datagram.channel_number}")
    print(f"Optional Header Length: {datagram.optional_header_length}")
    print(f"Selector Flags: {datagram.selector_flags}")
    print("--------------")

#main
DFPANLIST = []
root = tk.Tk()
root.withdraw()  # Hide the main window
    # Prompt the user to select a file
file_path = input("Enter the file path: ")
#remove inverted comma
# Open the file in read mode ('r' for reading)
if file_path.startswith('"') and file_path.endswith('"'):
    # Remove the enclosing double quotes
    file_path = file_path[1:-1]

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

    if index + 18 <= len(binary_data):
        extracted_bytes = binary_data[index:index + 18]
        unpacked_data = struct.unpack('>I H H H H I H',
                                      extracted_bytes)  # I refers to 4 bytes and H refers to 2 bytes for
        # print(f"Extracted Bytes: {extracted_bytes.hex()}")  # depict all 18 bytes that the EB200 header composes of
        magic_number = unpacked_data[0]
        version_minor = unpacked_data[1]
        version_major = unpacked_data[2]
        sequence_number = unpacked_data[3]
        seq_number_high = unpacked_data[4]
        data_size = unpacked_data[5]
        attributes = unpacked_data[6]
        if attributes ==1401:

            length = struct.unpack('>H', binary_data[index +16+2:index +16+2+ 2])[0]
            user_data = binary_data[index+16+2+ 2:index+16+2+ 2+ + length]
            numberoftracevalues = struct.unpack('>H', binary_data[index + 16 + 2 + 2  :index + 16 + 2 + 2 + 2])[0]
            channelnumber = struct.unpack('>B', binary_data[index + 16 + 2 + 2 +2:index + 16 + 2 + 2 + 2+1])[0]
            optionalheaderlength = struct.unpack('>B', binary_data[index + 16 + 2 + 2 +2+1:index + 16 + 2 + 2 +2+1+1])[0]
            selectorflags= struct.unpack('>I', binary_data[index + 16 + 2 + 2 +2+1+1:index + 16 + 2 + 2 +2+1+1+4])[0]
            datagram = EB200DatagramFormat(  # created an instance of the class EB200DatagramFormat
                magic_number,
                version_minor,
                version_major,
                sequence_number,
                seq_number_high,
                data_size,
                attributes,
                length,
                numberoftracevalues,
                channelnumber,
                optionalheaderlength,
                selectorflags
            )
            EB200headerprint()
            #start of optional header
            FreqLow=struct.unpack('<I', binary_data[index + 28:index + 28+ 4])[0]
            FreqHigh=struct.unpack('<I', binary_data[index + 28+4:index + 28+ 4+4])[0]
            FreqSpan = struct.unpack('<I', binary_data[index + 28 + 4 +4:index + 28 + 4 + 4+4])[0]
            DFThresholdMode = struct.unpack('<i', binary_data[index + 28 + 4 + 4+4:index + 28 + 4 + 4 + 4+4])[0]
            DFThresholdValue = struct.unpack('<i', binary_data[index + 28 + 4 + 4+4+4:index + 28 + 4 + 4 + 4+4+4])[0]
            DFBandwidth = struct.unpack('<I', binary_data[index + 28 + 4 + 4 + 4 + 4+4:index + 28 + 4 + 4 + 4 + 4 + 4+4])[0]
            StepWidth = struct.unpack('<I', binary_data[index + 28 + 4 + 4 + 4 + 4 + 4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4+4])[0]
            DFMeasureTime = struct.unpack('<I', binary_data[index + 28 + 4 + 4 + 4 + 4 + 4+4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4+4+4])[0]
            DFOption = struct.unpack('<i', binary_data[index + 28 + 4 + 4 + 4 + 4 + 4+4+4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4+4+4+4])[0]
            CompassHeading = struct.unpack('<H', binary_data[index + 28 + 4 + 4 + 4 + 4 + 4+4+4+4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4+4+4+4+2])[0]
            CompassHeadingType = struct.unpack('<h', binary_data[index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2+2])[0]
            AntennaFactor = struct.unpack('<i', binary_data[index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2+4])[0]
            DemodFreqChannel = struct.unpack('<i', binary_data[
                                                index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4+4])[0]
            DemodFreqLow = struct.unpack('<I', binary_data[
                                                   index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4+4])[
                0]
            DemodFreqHigh = struct.unpack('<I', binary_data[
                                               index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4])[0]
            OutputTimestamp = struct.unpack('<Q', binary_data[
                                               index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8])[
                0]
            Valid = struct.unpack('<h', binary_data[
                                                  index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8+2])[
                0]
            NoOfSatInView = struct.unpack('<h', binary_data[
                                        index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8+2+2])[0]
            LatRef = struct.unpack('<h', binary_data[
                                                index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8+2+2 :index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4+4+8+2+2+2])[0]
            LatDeg = struct.unpack('<h', binary_data[
                                         index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2])[
                0]
            LatMin = struct.unpack('<f', binary_data[
                                         index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2+2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4])[
                0]
            LonRef = struct.unpack('<h', binary_data[
                                         index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4+2])[
                0]
            LonDeg = struct.unpack('<h', binary_data[
                                         index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4+2+2])[
                0]
            LonMin = struct.unpack('<f', binary_data[
                                         index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4+2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4+2+2+4])[
                0]
            HDOP = struct.unpack('<f', binary_data[
                                         index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4+2+2+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 +2+4+2+2+4+4])[
                0]
            StepFreqNumerator = struct.unpack('<I', binary_data[
                index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4+4])[
                0]
            StepFreqDenominator = struct.unpack('<I', binary_data[
                                                    index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 +4])[
                0]
            DFBandwidthHighRes = struct.unpack('<Q', binary_data[
                                                     index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 +4+8])[
                0]
            Level = struct.unpack('<h', binary_data[
                                                     index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4+8:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8+2])[
                0]
            Azimuth = struct.unpack('<h', binary_data[
                                        index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2+2])[
                0]
            Quality = struct.unpack('<h', binary_data[
                                          index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2+2])[
                0]
            Elevation = struct.unpack('<h', binary_data[
                                          index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 +2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2+2])[
                0]
            Omniphase = struct.unpack('<h', binary_data[
                                          index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 +2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2+2+2])[
                0]
            Reserved = struct.unpack('<6B', binary_data[
                                            index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2+2:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2+6])[
                0]
            MeasureTimestamp = struct.unpack('<Q', binary_data[
                                            index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2+6:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8])[
                0]
            print("----DFPAN Optional Header----")
            print(f"Freq low: {FreqLow / 10 ** 6} MHz")
            print(f"Freq high: {FreqLow / 10 ** 6} MHz")
            print(f"Span: {FreqSpan/ 10 ** 6} MHz")
            print(f"Direction Finding: {DFThresholdMode}")
            print(f"Direction Finding Threshold Value: {DFThresholdValue} dBmicrovolt")
            print(f"DF Bandwidth: {DFBandwidth/ 10 **6} MHz")
            print(f"Step Width: {StepWidth/10**6} MHz")
            print(f"DF Measure Time: {DFMeasureTime/1000000} [s]")
            print(f"DFOption: {bin(DFOption)}")
            print(f"Comapass Heading: {CompassHeading}")
            print(f"Comapass Heading Type: {CompassHeadingType}")
            print(f"Antenna Factor: {AntennaFactor}")
            print(f"Demod Freq Channel: {DemodFreqChannel}")
            print(f"Demod Freq Low: {DemodFreqLow}")
            print(f"Demod Freq High: {DemodFreqHigh}")
            print("Output Time Stamp :",convert_unix_epoch(OutputTimestamp))
            print(f"Validity : {Valid}")
            print(f"No. of Satelites in View: {NoOfSatInView}")
            print(f"Lattiude Direction: {hex(LatRef)}")
            print(f"Lattiude Direction: {LatDeg} degrees")
            print(f"Geographical Lattitude in minutes: {LatMin} degrees")
            print(f"Longitude Direction: {hex(LonRef)}")
            print(f"Longitude Degrees: {LonMin}")
            print(f"Horizontal Dilution of Precision: {HDOP}")
            print(f"Step Frequency Numerator: {StepFreqNumerator}")
            print(f"Step Frequency Denominator: {StepFreqDenominator}")
            print(f"High-resolution represenation of current direction finding bandwidth: {DFBandwidthHighRes} mHz")
            print(f"Level : {Level}")
            print(f"Azimuth : {Azimuth}")
            print(f"Quality : {Quality}")
            print(f"Elevation : {Elevation}")
            print(f"Omniphase : {Omniphase}")
            print("Measured Time Stamp :", convert_unix_epoch(MeasureTimestamp))

            MeasureTimestamp = convert_unix_epoch(MeasureTimestamp)
            timestampdata.append(MeasureTimestamp)

            #     #store DF list andAzimuth List
            print(numberoftracevalues)
            format_string = f'<{numberoftracevalues}h'

            channel_data_size = numberoftracevalues * 2  # Assuming each value is 2 bytes (INT16)
            DFList=[]
            AzimuthList=[]
            data2=[]

            if index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8 + 4+channel_data_size <= len(binary_data):
                channel_values = struct.unpack(format_string, binary_data[
                                                              index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8+4:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8 +4+ channel_data_size])

                channel_data.append(channel_values)
            else:
                print("Not enough data to unpack channel data.")

            if index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8 + 4+2*channel_data_size <= len(binary_data):
                channel_values = struct.unpack(format_string, binary_data[
                                                              index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8+4+channel_data_size:index + 28 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 2 +6+8 +4+ 2*channel_data_size])

                channel_data2.append(channel_values)
            else:
                print("Not enough data to unpack channel data.")
    index += 1
if channel_data:
    print("Channel Data:")
    for i, values in enumerate(channel_data):
        print(f"DF_Level {i + 1}:{values}")
        #print(len(values))
        data.append(values)
#
if channel_data2:
    print("Channel Data:")
    for i, values in enumerate(channel_data2):
        print(f"Azimuth Values {i + 1}:{values}")
        #print(max(channel_data2[i]))
        #print(min(channel_data2[i]))
        data2.append(values)

updated_data = []
for index in range(len(data)):
    updated_index = tuple(value / 10 for value in data[index])
    updated_data.append(updated_index)

updated_data2 = []
for index in range(len(data2)):
    updated_index = tuple(value / 10 for value in data2[index])
    updated_data2.append(updated_index)
micro_symbol = '\u00B5'
# print(updated)
#FreqLow = FreqLow[0]
FreqLow = FreqLow / 10 ** 6
#FreqSpan = FreqSpan[0]
FreqSpan = FreqSpan / 10 ** 6

lower_frequency = FreqLow-FreqSpan/2
step_size = FreqSpan/numberoftracevalues
x = [(i * step_size) + lower_frequency for i in range(len(updated_data[i]))]
y= updated_data[index]
micro_symbol = '\u00B5'
import plotly.graph_objs as go
from dash import html, dash_table
import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
y=updated_data2[int(index)]
# Your clustering logic
data = {'X': x, 'Y': y}
df = pd.DataFrame(data)
epsilon = 4
min_samples = 16
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
df['Cluster'] = dbscan.fit_predict(df[['X', 'Y']])
cluster_counts = df['Cluster'].value_counts()
top_clusters = cluster_counts[cluster_counts.index != -1].nlargest(4)
top_points = df[df['Cluster'].isin(top_clusters.index)]
density_per_cluster = top_points.groupby('Cluster').size().sort_values(ascending=False)
colors = px.colors.qualitative.Plotly[:5]


micro_symbol = '\u00B5'
# print(updated)
# FreqLow = FreqLow / 10 ** 6
# #FreqSpan = FreqSpan[0]
# FreqSpan = FreqSpan / 10 ** 6

lower_frequency = FreqLow-FreqSpan/2
step_size = FreqSpan/numberoftracevalues
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

# Your existing data processing and calculations

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    dcc.Graph(id='plot'),
    dcc.Graph(id='plot2'),
    dbc.Alert(id='signal-alert', color='info', dismissable=False),
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

    html.Label('Epsilon value:'),
    dcc.Slider(
        id='epsilon-slider',
        min=0,
        max=5,
        step=0.1,
        value=4,
        marks={i: str(i) for i in range(6)},
        tooltip={'placement': 'bottom'}
    ),
    html.Label('Min samples:'),
    dcc.Slider(
        id='min-samples-slider',
        min=0,
        max=20,
        value=16,
        marks={i: str(i) for i in range(21)},
        tooltip={'placement': 'bottom'}
    ),
    html.Label('Number of density groups:'),
    dcc.Slider(
        id='density-groups-slider',
        min=2,
        max=15,
        value=5,
        marks={i: str(i) for i in range(2, 16)},
        tooltip={'placement': 'bottom'}
    ),
    dcc.Checklist(
        id='toggle-y-axis',
        options=[{'label': 'Fixed Y-Axis', 'value': 'fixed-y-axis'}],
        value=['fixed-y-axis']
    ),
    html.Label('Show Points Above Threshold Only:'),
    dcc.Checklist(
        id='show-above-threshold',
        options=[{'label': 'Show Points Above Threshold', 'value': 'show'}],
        value=[]
    ),
    html.Div([
        html.H2('Identified Signal Information'),
        dash_table.DataTable(
            id='bandwidth-table',
            columns=[
                {'name': 'Signal', 'id': 'Signal'},
                {'name': 'Bandwidth', 'id': 'Bandwidth'},
                {'name': 'Start Frequency', 'id': 'Start Frequency'},
                {'name': 'End Frequency', 'id': 'End Frequency'},
                {'name': 'Center Frequency', 'id': 'Center Frequency'},
                {'name': 'Median Signal Strength', 'id': 'Median Signal Strength'},
                {'name': 'Median Index', 'id': 'Median Index'},
                {'name': 'Angle of Arrival', 'id': 'Angle of Arrival'}
            ],
            style_table={'overflowX': 'auto'},
        )
    ]),
    html.Div(id='timestamp-output')
])

# Callback to update the plot based on sliders
@app.callback(
    [Output('plot', 'figure'), Output('plot2', 'figure'), Output('bandwidth-table', 'data'), Output('index-display', 'children'),Output('signal-alert', 'children')],
    [Input('index-slider', 'value'), Input('threshold-slider', 'value'), Input('toggle-y-axis', 'value'),
     Input('epsilon-slider', 'value'), Input('min-samples-slider', 'value'), Input('density-groups-slider', 'value'),Input('show-above-threshold', 'value')]
)
def update_plot(selected_index, threshold, toggle_value, epsilon, min_samples, num_density_groups,show_above_threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=updated_data[int(selected_index)], mode='lines+markers', name='Data'))

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
    #ML
    fig2 = go.Figure()


    # Adding layout details
    fig2.update_layout(
        xaxis_title='Frequency',
        yaxis_title='Angle of Arrival',
        title='Azimuth Level'
    )

    # Highlight points above the threshold in red
    above_threshold_x = [x[i] for i, y in enumerate(updated_data[int(selected_index)]) if y > threshold]
    above_threshold_y = [y for y in updated_data[int(selected_index)] if y > threshold]
    fig.add_trace(go.Scatter(x=above_threshold_x, y=above_threshold_y, mode='markers', marker=dict(color='red'),
                             name='Above Threshold'))

    # Identify individual signals based on index proximity
    bandwidth_data =[]
    signals = []
    bandwidths = []
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'pink', 'brown', 'lime', 'teal', 'magenta', 'red', 'yellow', 'olive']
 # Define colors for signals (add more if needed)
    color_idx = 0
    signal_indexes = []

    indexes_above_threshold = []

    if len(above_threshold_x) > 0:
        signal = [above_threshold_x[0]]
        signal_indexes = [i for i, y in enumerate(updated_data[int(selected_index)]) if y > threshold]
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
                median_index = int(np.median(signal_indexes)) if signal_indexes else None
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
    signal_counter = 0
    for idx, signal in enumerate(signals):
        if len(signal) > 1:
            signal_bandwidth = signal[-1] - signal[0]
            bandwidths.append(signal_bandwidth)
            median_index = int(np.median(signal_indexes)) if signal_indexes else None

            start_freq = signal[0]  # Start frequency of the signal
            end_freq = signal[-1]  # End frequency of the signal
            mean_freq = sum(signal) / len(signal)  # Mean frequency of the signal

            signal_points = []
            for i, x_val in enumerate(x):
                if x_val in signal:
                    signal_points.append((x_val, updated_data[int(selected_index)][i]))  # Collect x, y coordinates

            middle_point = None
            middle_point_index = None
            if signal_points:
                signal_points.sort()  # Sort the points by x value
                middle_point_index = len(signal_points) // 2  # Get the middle index
                middle_point = signal_points[middle_point_index][1]  # Extract the y-value of the middle point

                # Find the index of the middle point in the entire dataset
                middle_x_val = signal_points[middle_point_index][0]
                middle_point_index = x.index(middle_x_val)

                if middle_point_index is not None and middle_point_index < len(updated_data2[int(selected_index)]):
                    middle_point_value = updated_data2[int(selected_index)][middle_point_index]

            signal_counter += 1

            # Add data for each signal to bandwidth_data
            fig.update_layout(
                title='DFPAN Data ',
                xaxis_title='Frequencies (MHz)',
                yaxis_title=f'Signal Strength (db{micro_symbol})',
                annotations=[
                    dict(
                        text=f"Timestamp: {timestampdata[int(selected_index)]}",
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
            # indexes_str = ', '.join(str(idx) for idx in signal_indexes[idx])
            bandwidth_data.append({
                'Signal': f'Signal {signal_counter}',
                'Bandwidth': f'{signal_bandwidth:.2f} MHz',
                'Start Frequency': f'{start_freq:.2f} MHz',
                'End Frequency': f'{end_freq:.2f} MHz',
                'Center Frequency': f'{mean_freq:.2f} MHz',
                'Median Signal Strength': middle_point if middle_point else '',
                'Median Index': middle_point_index if middle_point_index is not None else '',
                'Angle of Arrival': middle_point_value if middle_point_value is not None else ''
                # Include the value from updated_data2 based on middle index
            })
    signal_color_map = {
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'yellow',
        5: 'purple',
        6: 'cyan',
        7: 'magenta',
        8: 'orange',
        9: 'lime',
        10: 'pink',
        11: 'teal',
        12: 'brown',
        13: 'olive',
        14: 'skyblue',
        15: 'indigo'
    }
    x_range = [min(x), max(x)]  # Define x-axis range

    fig.update_xaxes(range=x_range)  # Set x-axis range for fig
    indexes_above_threshold = [i for i, y in enumerate(updated_data[int(selected_index)]) if y > threshold]

    if 'show' in show_above_threshold:
        # Plot points from fig1 into fig2 with updated_data2
        # Plot points from fig1 into fig2 with updated_data2
        filtered_indexes_fig1 = []
        for idx in indexes_above_threshold:
            for signal in signals:
                if x[idx] in signal:
                    filtered_indexes_fig1.append(idx)
                    break  # Move to the next index

        x_values_fig2 = [x[idx] for idx in filtered_indexes_fig1]
        y_values_fig2 = [updated_data2[int(selected_index)][idx] for idx in filtered_indexes_fig1]

        signals_in_fig2 = []

        for idx in filtered_indexes_fig1:
            for signal_idx, signal in enumerate(signals):
                if x[idx] in signal:
                    signals_in_fig2.append(signal_idx + 1)
                    fig2.add_trace(go.Scatter(
                        x=signal,
                        y=[threshold] * len(signal),
                        mode='markers',
                        marker=dict(color=colors[signal_idx]),
                        showlegend=True
                    ))

        fig2.add_trace(go.Scatter(
            x=x_values_fig2,
            y=y_values_fig2,
            mode='markers',
            marker=dict(color='red'),
            name='Above Threshold'
        ))

        unique_signals_in_fig2 = list(set(signals_in_fig2))
        for signal_num in unique_signals_in_fig2:
            occurrences = signals_in_fig2.count(signal_num)
            if occurrences == 1:
                fig2.data = [trace for trace in fig2.data if trace.name != f'Signal {signal_num}']

        for i, (x_val, y_val) in enumerate(zip(x_values_fig2, y_values_fig2)):
            if signals_in_fig2[i] not in unique_signals_in_fig2:
                continue
            # fig2.add_annotation(
            #     x=x_val,
            #     y=y_val,
            #     text=f"Signal: {signals_in_fig2[i]}",
            #     showarrow=False,
            #     font=dict(color=colors[signals_in_fig2[i] - 1], size=12),
            #     xshift=5,
            #     yshift=5
            # )

        fig2.update_layout(
            legend=dict(
                title='Identified Angles',
                orientation='v',
                yanchor='auto',
                xanchor='auto'
            )
        )
        fig2.update_xaxes(range=x_range)
    else:
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(x=x,
                                  y=updated_data2[int(selected_index)],
                                  mode='markers',
                                  name='Azimuth Level'
                                  ))

        # Adding layout details
        fig2.update_layout(
            xaxis_title='Frequency',
            yaxis_title='Angle of Arrival',
            title='Azimuth Level'
        )
        # ML
        data = {'X': x, 'Y': updated_data2[int(selected_index)]}
        df = pd.DataFrame(data)
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        df['Cluster'] = dbscan.fit_predict(df[['X', 'Y']])

        cluster_counts = df['Cluster'].value_counts()
        top_clusters = cluster_counts[cluster_counts.index != -1].nlargest(num_density_groups)
        top_points = df[df['Cluster'].isin(top_clusters.index)]
        density_per_cluster = top_points.groupby('Cluster').size().sort_values(ascending=False)
        colors = px.colors.qualitative.Plotly[:num_density_groups]
        average_y_values = top_points.groupby('Cluster')['Y'].mean()

        fig2 = px.scatter(df, x='X', y='Y', color='Cluster', title=f'Azimuth Level')
        fig2.update_traces(marker=dict(size=8, color='rgba(0, 0, 0, 0.3)'))
        fig2.update_xaxes(title_text='Frequency (MHz)')
        fig2.update_yaxes(title_text='Angle of Arrival')
        fig2.update_xaxes(range=x_range)  # Set x-axis range for fig2

        for i, cluster_id in enumerate(top_clusters.index):
            cluster_points = top_points[top_points['Cluster'] == cluster_id]
            fig2.add_trace(px.scatter(cluster_points, x='X', y='Y').data[0])
            fig2.data[i + 1].marker.color = colors[i]

            # Label each cluster with its density ranking
            fig2.add_annotation(
                x=cluster_points['X'].mean(),
                y=cluster_points['Y'].max() + 0.35 * (cluster_points['Y'].max() - cluster_points['Y'].min()),
                # Adjust the position
                text=f"Density Rank: {i + 1}<br>Points: {density_per_cluster[cluster_id]}<br>Avg AoA: {average_y_values[cluster_id]:.2f}",
                showarrow=False,
                font=dict(color="black", size=10),
                xanchor="center",
                yanchor="bottom"
            )



    yaxis_settings = {}
    if 'fixed-y-axis' in toggle_value:
        yaxis_settings['fixedrange'] = True
    else:
        yaxis_settings['fixedrange'] = False

    if yaxis_settings.get('fixedrange'):
        yaxis_settings['range'] = [min_data, max_data]  # Define your custom range here

    fig.update_layout(yaxis=yaxis_settings)
    # fig2.update_layout(yaxis=yaxis_settings)

    count = len([y for y in updated_data[selected_index] if y > threshold])

    alert_message = dbc.Alert(f'There is/are {signal_counter} identified signal/signals above the threshold amounting to {count} data points.', color='info',
                              dismissable=True) if count > 0 else None

    return fig,fig2, bandwidth_data, index_display, alert_message

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8007, debug=False)