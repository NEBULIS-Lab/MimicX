"""CPU-only contracts for the optional, logging-only SONIC C++ patch.

The small C++ fixture models the released buffer/logger interfaces, not the
policy or DDS runtime. No vendor checkout, robot assets or GPU is required.
"""
import csv
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

from mimicx.experiments.sonic_helpers import instrument_startup_cpp


READ = "      auto imu_data = imu_torso_buffer_.GetDataWithTime();"
CALL = "        state_logger_->LogFullState(base_quat,"
END = "                                    ros_timestamp);\n      }\n      return true;\n    }"

# Use immutable buffer snapshots and a logger that prints every input so the
# original and patched executables can be compared without fitting their clocks.
CPP = r'''
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
using Clock = std::chrono::steady_clock;
struct LowState {
  uint32_t tick() const { return 2000; }
  uint8_t mode_machine() const { return 128; }
};
struct Imu {
  const std::array<float, 3>& gyroscope() const { return gyro; }
  const std::array<float, 3>& accelerometer() const { return accel; }
  std::array<float, 3> gyro{.125f, -.25f, .5f}, accel{.25f, .5f, 9.75f};
};
template <typename T> struct Snapshot {
  std::shared_ptr<const T> data;
  Clock::time_point timestamp;
};
template <typename T> struct Buffer {
  Snapshot<T> snapshot;
  int reads = 0;
  Snapshot<T> GetDataWithTime() { ++reads; return snapshot; }
};
struct Logger {
  uint64_t next = 0;
  uint64_t LogFullState(double base, double q, double dq, double action,
                        double simulation_time, double policy_time,
                        double ros_timestamp) {
    std::cout << "log " << next << " " << base << " " << q << " " << dq
              << " " << action << " " << simulation_time << " " << policy_time
              << " " << ros_timestamp << "\n";
    return next++;
  }
};
struct Deploy {
  Buffer<LowState> low_state_buffer_{{std::make_shared<LowState>(),
      Clock::time_point(std::chrono::milliseconds(12125))}};
  Buffer<Imu> imu_torso_buffer_{{std::make_shared<Imu>(),
      Clock::time_point(std::chrono::milliseconds(12500))}};
  Logger logger;
  Logger* state_logger_ = &logger;
  double q = .25, dq = -.5, action = 0., sim_time = 2., policy_time = 3.;
  bool GatherRobotStateToLogger() {
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      auto imu_data = imu_torso_buffer_.GetDataWithTime();
      const auto ls = low_state_data.data;
      const auto imu_torso = imu_data.data;
      if (!ls || !imu_torso) return false;
      action = q + 2. * dq;
      const double base_quat = 1.;
      if (state_logger_) {
        const double ros_timestamp = 4.125;
        state_logger_->LogFullState(base_quat, q, dq, action, sim_time, policy_time,
                                    ros_timestamp);
      }
      return true;
    }
};
int main(int argc, char** argv) {
  Deploy deploy;
  const std::string mode = argc > 1 ? argv[1] : "normal";
  if (mode == "no_logger") deploy.state_logger_ = nullptr;
  if (mode == "no_lowstate") deploy.low_state_buffer_.snapshot.data.reset();
  if (mode == "no_torso") deploy.imu_torso_buffer_.snapshot.data.reset();
  if (mode == "nonzero_index") deploy.logger.next = 7;
  for (int step = 0; step < 2; ++step) {
    const bool result = deploy.GatherRobotStateToLogger();
    std::cout << "control " << result << " " << deploy.action << " " << deploy.q
              << " " << deploy.dq << " " << deploy.sim_time << " " << deploy.policy_time
              << " " << deploy.low_state_buffer_.reads << " " << deploy.imu_torso_buffer_.reads
              << " " << deploy.logger.next << "\n";
    deploy.q += .125;
    deploy.sim_time += .005;
    deploy.policy_time += .02;
    auto next_imu = std::make_shared<Imu>();
    next_imu->gyro[0] = 99.f;
    if (mode != "no_torso") deploy.imu_torso_buffer_.snapshot.data = next_imu;
  }
}
'''


class StartupInstrumentationSourceTests(unittest.TestCase):
    def test_only_logging_region_and_read_timestamp_change(self):
        patched = instrument_startup_cpp(CPP)
        includes = "#include <fstream>\n#include <iomanip>\n#include <cstdlib>\n"
        self.assertTrue(patched.startswith(includes))
        restored = patched[len(includes):]
        timestamp = "\n      const auto mimicx_control_read = std::chrono::steady_clock::now();"
        self.assertEqual(restored.count(timestamp), 1)
        self.assertIn(READ + timestamp, restored)
        restored = restored.replace(timestamp, "")
        captured_call = "        auto mimicx_state_index = state_logger_->LogFullState(base_quat,"
        self.assertEqual(restored.count(captured_call), 1)
        restored = restored.replace(captured_call, CALL)
        start = restored.index("\n        if (mimicx_state_index == 0) {")
        stop = restored.index("      }\n      return true;\n    }", start)
        logging_block = restored[start:stop]
        self.assertIn('std::getenv("MIMICX_STARTUP_STATE_LOG")', logging_block)
        self.assertIn("std::setprecision(9)", logging_block)
        self.assertNotIn("GetDataWithTime", logging_block)
        self.assertNotIn("steady_clock::now", logging_block)
        self.assertEqual(restored[:start] + "\n" + restored[stop:], CPP)

    def test_missing_or_duplicate_anchors_fail_closed(self):
        for anchor in (READ, CALL, END):
            for changed in (CPP.replace(anchor, "", 1), CPP + "\n" + anchor):
                with self.subTest(anchor=anchor, duplicate=changed.startswith(CPP)):
                    with self.assertRaisesRegex(ValueError, "startup logger anchors differ"):
                        instrument_startup_cpp(changed)

    def test_deterministic_and_rejects_double_instrumentation(self):
        original = CPP[:]
        patched = instrument_startup_cpp(CPP)
        self.assertEqual(CPP, original)
        self.assertEqual(patched, instrument_startup_cpp(CPP))
        with self.assertRaisesRegex(ValueError, "startup logger anchors differ"):
            instrument_startup_cpp(patched)


class StartupInstrumentationExecutableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("C++ compiler required for executable instrumentation checks")
        cls.temporary = tempfile.TemporaryDirectory(prefix="sonic-startup-cpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        cls.environment.pop("MIMICX_STARTUP_STATE_LOG", None)
        cls.executables = []
        for name, source in (("original", CPP), ("instrumented", instrument_startup_cpp(CPP))):
            cpp = cls.root / (name + ".cpp")
            cpp.write_text(source)
            binary = cls.root / name
            result = subprocess.run(
                [*compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)],
                env=cls.environment, capture_output=True, text=True, timeout=60,
            )
            if result.returncode:
                raise AssertionError("CPU fixture compilation failed:\n" + result.stdout + result.stderr)
            cls.executables.append(binary)

    def run_pair(self, *, mode="normal", log_path=None):
        environment = dict(self.environment)
        if log_path is not None:
            environment["MIMICX_STARTUP_STATE_LOG"] = str(log_path)
        results = []
        for binary in self.executables:
            result = subprocess.run([str(binary), mode], env=environment,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            results.append(result.stdout)
            if binary == self.executables[0] and log_path is not None:
                self.assertFalse(log_path.exists())
        self.assertEqual(results[0], results[1], "instrumentation changed control/logger behavior")
        return results[1]

    def test_enabled_sidecar_preserves_calls_and_records_first_snapshot_only(self):
        path = self.root / "first-source.csv"
        before = time.monotonic()
        output = self.run_pair(log_path=path)
        after = time.monotonic()
        self.assertEqual(output.count("log "), 2)
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, [
                "index", "control_read_monotonic", "lowstate_received_monotonic",
                "torso_received_monotonic", "lowstate_tick", "mode_machine",
                *[f"torso_gyro_{i}" for i in range(3)],
                *[f"torso_accel_{i}" for i in range(3)],
            ])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["index"], "0")
        self.assertEqual(row["lowstate_tick"], "2000")
        self.assertEqual(row["mode_machine"], "128")
        self.assertEqual(row["lowstate_received_monotonic"], "12.125000000")
        self.assertEqual(row["torso_received_monotonic"], "12.500000000")
        self.assertGreaterEqual(float(row["control_read_monotonic"]), before)
        self.assertLessEqual(float(row["control_read_monotonic"]), after)
        for prefix, values in (("gyro", [.125, -.25, .5]), ("accel", [.25, .5, 9.75])):
            self.assertEqual([float(row[f"torso_{prefix}_{i}"]) for i in range(3)], values)

    def test_unset_environment_preserves_existing_logging(self):
        self.assertEqual(self.run_pair().count("log "), 2)

    def test_missing_inputs_disabled_logger_and_later_rows_do_not_write(self):
        for mode in ("no_lowstate", "no_torso", "no_logger", "nonzero_index"):
            with self.subTest(mode=mode):
                path = self.root / (mode + ".csv")
                self.run_pair(mode=mode, log_path=path)
                self.assertFalse(path.exists())

    def test_unwritable_destination_does_not_change_control_behavior(self):
        # Missing parent is deterministic even for a privileged test process.
        path = self.root / "missing-parent" / "source.csv"
        self.run_pair(log_path=path)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
