#!/usr/bin/env ruby
#
# Script to generate a testing bathymetry file in CDL format
#     ruby bathymetry.rb > bathymetry.cdl  
# 
# The output format is not very nicely formatted, so it should
# be converted to NetCDF using
#   ncgen -o test_bathymetry.nc bathymetry.cdl
# and then re-converted to CDL using
#   ncdump test_bathymetry.nc > test_bathymetry.cdl
#

x = []
y = []
z = []

-245.upto(745) {|t| x << t if (t-5) % 10 == 0}
-1225.upto(1225) {|t| y << t if (t-25) % 50 == 0}

y.each do |yc|
  x.each do |xc|
    z << (xc*yc)/1000.0
  end
end  

tmpl = <<TMPL
netcdf test_bathymetry {
dimensions:
        x = #{x.length} ;
        y = #{y.length} ;
variables:
        float x(x) ;
        float y(y) ;
        float z(y, x) ;

// global attributes:
                :Conventions = "COARDS" ;
data:
 x = #{x.join ","};
 y = #{y.join ","};
 z = #{z.join ","};
}
TMPL

puts tmpl
