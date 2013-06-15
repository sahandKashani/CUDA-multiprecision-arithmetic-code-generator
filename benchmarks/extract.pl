#!/usr/bin/perl -w
use strict;

# Global Data ==================================================================
my $file_name = 'data';
my @function_names = ('add_loc_kernel', 'add_glo_kernel', 'sub_loc_kernel', 'sub_glo_kernel', 'mul_loc_kernel', 'mul_glo_kernel', 'mul_karatsuba_loc_kernel', 'mul_karatsuba_glo_kernel', 'add_m_loc_kernel', 'add_m_glo_kernel', 'sub_m_loc_kernel', 'sub_m_glo_kernel', 'add_loc_assembly', 'add_loc_C');
my %index_map = (
  'Start' => 1,
  'Duration' => 2,
  'Grid Size' => 3,
  'Block Size' => 4,
  'Regs*' => 5,
  'SSMem*' => 6,
  'DSMem*' => 7,
  'Size' => 8,
  'Throughput' => 9,
  'Device' => 10,
  'Context' => 11,
  'Stream' => 12,
  'Name' => 13
);
my $grid_size_limit = 10;
my $block_size_limit = 10;
my $field_to_extract = shift;
# ==============================================================================

# Functions ====================================================================
sub log2 {
  my $n = shift;
  return log($n)/log(2);
}

sub agglomerate_data {
  open RESULTS, "> $file_name";
  while (<>) {
    if ( / (^\s{2}) \d\.\d\d /x ) {
      print RESULTS $_ if ( not /CUDA/ );
    }
  }
  close RESULTS;
}

sub assemble_data {
  my $function_name = $_[0];
  my $field_to_extract = $_[1];
  my @extracted_information;

  open RESULTS, "> $function_name";
  open DATA, "< $file_name";
  while (<DATA>) {
    if ( /${function_name}/ ) {
      my @line_fields = split /\s{2,}/;

      my $grid_size = $line_fields[$index_map{'Grid Size'}];
      $grid_size =~ s/\(//;
      $grid_size =~ s/\)//;
      my @grid_size = split /\s/, $grid_size;

      my $block_size = $line_fields[$index_map{'Block Size'}];
      $block_size =~ s/\(//;
      $block_size =~ s/\)//;
      my @block_size = split /\s/, $block_size;

      $extracted_information[$grid_size_limit * log2($grid_size[0]) + log2($block_size[0])] = $line_fields[$index_map{$field_to_extract}];
    }
  }
  close DATA;
  foreach (@extracted_information) {
    print RESULTS "$_\n" if (defined);
  }
  close RESULTS;
}

sub extract_data {
  unlink $file_name;
  unlink foreach (@function_names);
  agglomerate_data;
  assemble_data( $_, $field_to_extract ) foreach (@function_names);
  unlink $file_name;
}
# ==============================================================================

extract_data;
