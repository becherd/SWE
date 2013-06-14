#
# Copyright (c) 2010 Western Digital Corporation
# Alan Somers asomers (at) gmail (dot) com
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import inspect

import SCons.Script
import SCons.Taskmaster

class RightNowMixin:
    """Dynamic base class that adds to SCons Environment classes"""
    def RightNow(self, targets):
        """Builds targets Right Now.

        Checks to see whether any of targets are out of date, and if so builds
        them before returning.  If they are built right now, then the output
        files will be available for processing when this function returns, and
        they will not be rebuilt during the main build phase (unless their
        dependencies have changed by then)
        """
        for node in targets:
            node.disambiguate()
            if not node.is_up_to_date():
                #We always leave the dependencies in order, though in theory
                #there's no reason that we couldn't shuffle them
                def order(x): return x
                options = None
                #Here we find the command-line options.  They weren't really
                #intended to be accessible from our SConscripts, so we have
                #to dig them out of the stack.  This interface is unofficial
                #and may break with future SCons releases
                for stack in inspect.stack():
                    frame = stack[0]
                    if 'options' in frame.f_locals:
                        options = frame.f_locals['options']
                        SCons.Script.Main.BuildTask.options = options
                assert options != None, "'options' not found on the stack!"
                if options.taskmastertrace_file == '-':
                    tmtrace = sys.stdout
                elif options.taskmastertrace_file:
                    tmtrace = open(options.taskmastertrace_file, 'wb')
                else:
                    tmtrace = None
                taskmaster = SCons.Taskmaster.Taskmaster(targets, 
                                SCons.Script.Main.BuildTask, order, tmtrace)
                num_jobs = options.num_jobs
                jobs = SCons.Job.Jobs(num_jobs, taskmaster)
                jobs.run()
                break   #if we build one node, we build them all so break now

def exists(env):
    """This tool requires no support from the OS; it is always available"""
    return True

def generate(env):
    """Add RightNow to SCons.

    Since RightNow is a mixin, it will be available in all environments"""
    env.__class__.__bases__ += (RightNowMixin,)

