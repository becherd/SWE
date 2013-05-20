#!/usr/bin/env ruby
#
# Script to generate a testing displacement file in CDL format
#     ruby displacement.rb > displacement.cdl  
# 
# The output format is not very nicely formatted, so it should
# be converted to NetCDF using
#   ncgen -o test_displacement.nc displacement.cdl
# and then re-converted to CDL using
#   ncdump test_displacement.nc > test_displacement.cdl
#

x = []
y = []
z = []

155.upto(345) {|t| x << t if (t-5) % 10 == 0}
-450.upto(450) {|t| y << t if (t-50) % 100 == 0}

y.each do |yc|
  x.each do |xc|
    z << (xc+yc)/2.0
  end
end  

tmpl = <<TMPL
netcdf test_displacement {
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
