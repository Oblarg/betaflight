#
# WhippyFlight (OPTIONS=USE_WHIPPYFLIGHT): build the Rust controller
# staticlib with cargo and link it.  Included after the MCU .mk, which sets
# ARCH_FLAGS; builds without the option never read this file or need Rust.
#
# Needs cargo and the target's standard library (rustup target add <triple>).
# Cargo tracks the Rust sources itself, so the library rule always runs it:
# a no-op build leaves the library untouched and the firmware isn't relinked.
#

WHIPPYFLIGHT_CRATE     := $(ROOT)/src/main/flight/whippyflight
WHIPPYFLIGHT_CARGO_DIR := $(OBJECT_DIR)/whippyflight
CARGO                  ?= cargo

ifeq ($(shell command -v $(CARGO)),)
$(error USE_WHIPPYFLIGHT needs Rust's cargo (https://rustup.rs))
endif

# The Rust target matching the MCU's core and float ABI.
ifeq ($(TARGET_MCU_FAMILY),SITL)
WHIPPYFLIGHT_TARGET ?= $(shell rustc -vV | sed -n 's/^host: //p')
else ifneq ($(filter -mfloat-abi=hard,$(ARCH_FLAGS)),)
ifneq ($(filter -mcpu=cortex-m4 -mcpu=cortex-m7,$(ARCH_FLAGS)),)
WHIPPYFLIGHT_TARGET ?= thumbv7em-none-eabihf
else ifneq ($(filter -mcpu=cortex-m33 -mcpu=cortex-m55,$(ARCH_FLAGS)),)
WHIPPYFLIGHT_TARGET ?= thumbv8m.main-none-eabihf
endif
endif
ifeq ($(WHIPPYFLIGHT_TARGET),)
$(error USE_WHIPPYFLIGHT: no Rust target for $(TARGET_MCU_FAMILY) ($(ARCH_FLAGS)); it needs a hard-float Cortex-M4/M7/M33/M55 or SITL)
endif

WHIPPYFLIGHT_LIB := $(WHIPPYFLIGHT_CARGO_DIR)/$(WHIPPYFLIGHT_TARGET)/release/libwhippyflight_betaflight.a
EXTRA_LD_FLAGS   += $(WHIPPYFLIGHT_LIB)

$(WHIPPYFLIGHT_LIB): FORCE
	@echo "Building WhippyFlight for $(WHIPPYFLIGHT_TARGET)" "$(STDOUT)"
	$(V1) $(CARGO) build --release --locked --quiet \
		--manifest-path $(WHIPPYFLIGHT_CRATE)/Cargo.toml \
		--target $(WHIPPYFLIGHT_TARGET) \
		--target-dir $(WHIPPYFLIGHT_CARGO_DIR)

.PHONY: FORCE
FORCE:
