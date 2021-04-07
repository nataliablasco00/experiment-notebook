import os
import sys
import subprocess
import tempfile
from enb import icompression
from enb.config import get_options
from enb import tcall

options = get_options()


class Kakadu(icompression.WrapperCodec, icompression.LosslessCodec, icompression.LossyCodec):
    """TODO: add docstring for the classes, the module and non-inherited methods.
    """

    def __init__(self, ht=False, spatial_dwt_levels=5, lossless=None, bit_rate=False, quality_factor=False):
        assert isinstance(ht, bool), "HT must be a boolean (True/False)"
        assert spatial_dwt_levels in range(0, 34)
        if lossless:
            assert bit_rate is False, "a bit rate can not be set if lossless is True"
            assert quality_factor is False, "a quality factor can not be set if lossless is True"
        elif lossless is None or not lossless:
            if bit_rate:
                assert bit_rate > 0
                lossless = False
            if quality_factor:
                assert 0 < quality_factor <= 100
                lossless = False
        else:
            lossless = True

        icompression.WrapperCodec.__init__(
            self,
            compressor_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kdu_compress"),
            decompressor_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kdu_expand"),
            param_dict=dict(
                ht=ht,
                spatial_dwt_levels=spatial_dwt_levels,
                lossless=lossless,
                bit_rate=bit_rate,
                quality_factor=quality_factor))

    def get_compression_params(self, original_path, compressed_path, original_file_info):
        return f"-i {original_path}*{original_file_info['component_count']}" \
               f"@{original_file_info['width'] * original_file_info['height'] * original_file_info['bytes_per_sample']} " \
               f"-o {compressed_path} -no_info -full -no_weights " \
               f"Corder=LRCP " \
               f"Clevels={self.param_dict['spatial_dwt_levels']} " \
               f"Clayers={original_file_info['component_count']} " \
               f"Creversible={'yes' if self.param_dict['lossless'] else 'no'} " \
               f"Cycc=no " \
               f"Sdims=\\{{{original_file_info['width']},{original_file_info['height']}\\}} " \
               f"Nprecision={original_file_info['bytes_per_sample'] * 8} " \
               f"Sprecision={original_file_info['bytes_per_sample'] * 8} " \
               f"Nsigned={'yes' if original_file_info['signed'] else 'no'} " \
               f"Ssigned={'yes' if original_file_info['signed'] else 'no'} " \
               f"{'Cmodes=HT' if self.param_dict['ht'] else ''} " \
               f"{'-rate ' + str(self.param_dict['bit_rate']) if self.param_dict['bit_rate'] else ''} " \
               f"{'Qfactor=' + str(self.param_dict['quality_factor']) if self.param_dict['quality_factor'] else ''}"

    def get_decompression_params(self, compressed_path, reconstructed_path, original_file_info):
        return f"-i {compressed_path} -o {reconstructed_path} -raw_components"

    def decompress(self, compressed_path, reconstructed_path, original_file_info=None):
        temp_list = []
        temp_path = f""
        for i in range(0, original_file_info['component_count']):
            temp_list.append(tempfile.NamedTemporaryFile(suffix=".raw").name)
            if i < (original_file_info['component_count'] - 1):
                temp_path += f"{temp_list[i]},"
            else:
                temp_path += f"{temp_list[i]}"

        decompression_results = icompression.WrapperCodec.decompress(
            self, compressed_path, reconstructed_path=temp_path, original_file_info=original_file_info)

        with open(reconstructed_path, "wb") as output_file:
            for p in temp_path.split(","):
                with open(p, "rb") as component_file:
                    output_file.write(component_file.read())
        decompression_results.reconstructed_path = reconstructed_path

        return decompression_results

    @property
    def label(self):
        return f"Kakadu {'HT' if self.param_dict['ht'] else ''}" \
               f" {'lossless' if self.param_dict['lossless'] else 'lossy'}"


# TODO: the bitrate does not work well: please read the help for    -rate -|<bits/pel>,<bits/pel>,... in kdu_compress
# and fix

# TODO: Kakadu MCT does not accept quality factor not bitrate - it totally should accept and pass
# those to the parent initializator

# TODO: add different parameters to kakadu MCT in the lossy compression experiment
class Kakadu_MCT(Kakadu):
    def __init__(self, ht=False, spatial_dwt_levels=5, spectral_dwt_levels=5, lossless=None):
        assert 0 <= spectral_dwt_levels <= 32, f"Invalid number of spectral levels"
        Kakadu.__init__(self, ht=ht, spatial_dwt_levels=spatial_dwt_levels, lossless=lossless)
        self.param_dict["spectral_dwt_levels"] = spectral_dwt_levels

    def get_compression_params(self, original_path, compressed_path, original_file_info):
        # TODO: split into 2D and 3D transform arguments, put here only the ones that do not
        # appear for the 2D case
        return Kakadu.get_compression_params(
            self,
            original_path=original_path,
            compressed_path=compressed_path,
            original_file_info=original_file_info) + \
               f" Mcomponents={original_file_info['component_count']} " \
               f"Mstage_inputs:I1=\\{{0,{original_file_info['component_count'] - 1}\\}} " \
               f"Mstage_outputs:I1=\\{{0,{original_file_info['component_count'] - 1}\\}} " \
               f"Mstage_collections:I1=\\{{{original_file_info['component_count']},{original_file_info['component_count']}\\}} " \
               f"Mstage_xforms:I1=\\{{DWT," \
               + ('1' if self.param_dict['lossless'] else '0') + \
               f",4,0,{self.param_dict['spectral_dwt_levels']}\\}} " \
               f"Mvector_size:I4={original_file_info['component_count']} " \
               f"Mvector_coeffs:I4=0 Mnum_stages=1 Mstages=1"

    def get_decompression_params(self, compressed_path, reconstructed_path, original_file_info):
        return f"-i {compressed_path} -o {reconstructed_path} -raw_components "

    @property
    def label(self):
        return f"Kakadu MCT {'HT' if self.param_dict['ht'] else ''}" \
               f" {'lossless' if self.param_dict['lossless'] else 'lossy'}"
