// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: rdf.proto

#ifndef PROTOBUF_rdf_2eproto__INCLUDED
#define PROTOBUF_rdf_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2006000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace rdf {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_rdf_2eproto();
void protobuf_AssignDesc_rdf_2eproto();
void protobuf_ShutdownFile_rdf_2eproto();

class RDFParameter;

enum RDFParameter_Col {
  RDFParameter_Col_RED = 0,
  RDFParameter_Col_GREEN = 1,
  RDFParameter_Col_BLUE = 2,
  RDFParameter_Col_BLACK = 3,
  RDFParameter_Col_WHITE = 4,
  RDFParameter_Col_YELLOW = 5,
  RDFParameter_Col_PURPLE = 6,
  RDFParameter_Col_AZURE = 7
};
bool RDFParameter_Col_IsValid(int value);
const RDFParameter_Col RDFParameter_Col_Col_MIN = RDFParameter_Col_RED;
const RDFParameter_Col RDFParameter_Col_Col_MAX = RDFParameter_Col_AZURE;
const int RDFParameter_Col_Col_ARRAYSIZE = RDFParameter_Col_Col_MAX + 1;

const ::google::protobuf::EnumDescriptor* RDFParameter_Col_descriptor();
inline const ::std::string& RDFParameter_Col_Name(RDFParameter_Col value) {
  return ::google::protobuf::internal::NameOfEnum(
    RDFParameter_Col_descriptor(), value);
}
inline bool RDFParameter_Col_Parse(
    const ::std::string& name, RDFParameter_Col* value) {
  return ::google::protobuf::internal::ParseNamedEnum<RDFParameter_Col>(
    RDFParameter_Col_descriptor(), name, value);
}
// ===================================================================

class RDFParameter : public ::google::protobuf::Message {
 public:
  RDFParameter();
  virtual ~RDFParameter();

  RDFParameter(const RDFParameter& from);

  inline RDFParameter& operator=(const RDFParameter& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const RDFParameter& default_instance();

  void Swap(RDFParameter* other);

  // implements Message ----------------------------------------------

  RDFParameter* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const RDFParameter& from);
  void MergeFrom(const RDFParameter& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  typedef RDFParameter_Col Col;
  static const Col RED = RDFParameter_Col_RED;
  static const Col GREEN = RDFParameter_Col_GREEN;
  static const Col BLUE = RDFParameter_Col_BLUE;
  static const Col BLACK = RDFParameter_Col_BLACK;
  static const Col WHITE = RDFParameter_Col_WHITE;
  static const Col YELLOW = RDFParameter_Col_YELLOW;
  static const Col PURPLE = RDFParameter_Col_PURPLE;
  static const Col AZURE = RDFParameter_Col_AZURE;
  static inline bool Col_IsValid(int value) {
    return RDFParameter_Col_IsValid(value);
  }
  static const Col Col_MIN =
    RDFParameter_Col_Col_MIN;
  static const Col Col_MAX =
    RDFParameter_Col_Col_MAX;
  static const int Col_ARRAYSIZE =
    RDFParameter_Col_Col_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  Col_descriptor() {
    return RDFParameter_Col_descriptor();
  }
  static inline const ::std::string& Col_Name(Col value) {
    return RDFParameter_Col_Name(value);
  }
  static inline bool Col_Parse(const ::std::string& name,
      Col* value) {
    return RDFParameter_Col_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // optional int32 num_images = 1 [default = 10000];
  inline bool has_num_images() const;
  inline void clear_num_images();
  static const int kNumImagesFieldNumber = 1;
  inline ::google::protobuf::int32 num_images() const;
  inline void set_num_images(::google::protobuf::int32 value);

  // optional int32 num_samples = 2 [default = 2000];
  inline bool has_num_samples() const;
  inline void clear_num_samples();
  static const int kNumSamplesFieldNumber = 2;
  inline ::google::protobuf::int32 num_samples() const;
  inline void set_num_samples(::google::protobuf::int32 value);

  // repeated int32 num_pixels = 3;
  inline int num_pixels_size() const;
  inline void clear_num_pixels();
  static const int kNumPixelsFieldNumber = 3;
  inline ::google::protobuf::int32 num_pixels(int index) const;
  inline void set_num_pixels(int index, ::google::protobuf::int32 value);
  inline void add_num_pixels(::google::protobuf::int32 value);
  inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      num_pixels() const;
  inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_num_pixels();

  // repeated .rdf.RDFParameter.Col color = 4;
  inline int color_size() const;
  inline void clear_color();
  static const int kColorFieldNumber = 4;
  inline ::rdf::RDFParameter_Col color(int index) const;
  inline void set_color(int index, ::rdf::RDFParameter_Col value);
  inline void add_color(::rdf::RDFParameter_Col value);
  inline const ::google::protobuf::RepeatedField<int>& color() const;
  inline ::google::protobuf::RepeatedField<int>* mutable_color();

  // optional int32 num_trees = 5 [default = 10];
  inline bool has_num_trees() const;
  inline void clear_num_trees();
  static const int kNumTreesFieldNumber = 5;
  inline ::google::protobuf::int32 num_trees() const;
  inline void set_num_trees(::google::protobuf::int32 value);

  // optional int32 num_depth = 6 [default = 20];
  inline bool has_num_depth() const;
  inline void clear_num_depth();
  static const int kNumDepthFieldNumber = 6;
  inline ::google::protobuf::int32 num_depth() const;
  inline void set_num_depth(::google::protobuf::int32 value);

  // optional int32 max_span = 7 [default = 200];
  inline bool has_max_span() const;
  inline void clear_max_span();
  static const int kMaxSpanFieldNumber = 7;
  inline ::google::protobuf::int32 max_span() const;
  inline void set_max_span(::google::protobuf::int32 value);

  // optional int32 space_size = 8 [default = 400];
  inline bool has_space_size() const;
  inline void clear_space_size();
  static const int kSpaceSizeFieldNumber = 8;
  inline ::google::protobuf::int32 space_size() const;
  inline void set_space_size(::google::protobuf::int32 value);

  // optional int32 num_features = 9 [default = 2000];
  inline bool has_num_features() const;
  inline void clear_num_features();
  static const int kNumFeaturesFieldNumber = 9;
  inline ::google::protobuf::int32 num_features() const;
  inline void set_num_features(::google::protobuf::int32 value);

  // optional int32 num_thresholds = 10 [default = 50];
  inline bool has_num_thresholds() const;
  inline void clear_num_thresholds();
  static const int kNumThresholdsFieldNumber = 10;
  inline ::google::protobuf::int32 num_thresholds() const;
  inline void set_num_thresholds(::google::protobuf::int32 value);

  // optional string input_path = 11 [default = "./data"];
  inline bool has_input_path() const;
  inline void clear_input_path();
  static const int kInputPathFieldNumber = 11;
  inline const ::std::string& input_path() const;
  inline void set_input_path(const ::std::string& value);
  inline void set_input_path(const char* value);
  inline void set_input_path(const char* value, size_t size);
  inline ::std::string* mutable_input_path();
  inline ::std::string* release_input_path();
  inline void set_allocated_input_path(::std::string* input_path);

  // optional string output_path = 12 [default = "./new_data"];
  inline bool has_output_path() const;
  inline void clear_output_path();
  static const int kOutputPathFieldNumber = 12;
  inline const ::std::string& output_path() const;
  inline void set_output_path(const ::std::string& value);
  inline void set_output_path(const char* value);
  inline void set_output_path(const char* value, size_t size);
  inline ::std::string* mutable_output_path();
  inline ::std::string* release_output_path();
  inline void set_allocated_output_path(::std::string* output_path);

  // optional string out_name = 13 [default = "./forest.txt"];
  inline bool has_out_name() const;
  inline void clear_out_name();
  static const int kOutNameFieldNumber = 13;
  inline const ::std::string& out_name() const;
  inline void set_out_name(const ::std::string& value);
  inline void set_out_name(const char* value);
  inline void set_out_name(const char* value, size_t size);
  inline ::std::string* mutable_out_name();
  inline ::std::string* release_out_name();
  inline void set_allocated_out_name(::std::string* out_name);

  // optional string test_input_path = 14 [default = "./test_data"];
  inline bool has_test_input_path() const;
  inline void clear_test_input_path();
  static const int kTestInputPathFieldNumber = 14;
  inline const ::std::string& test_input_path() const;
  inline void set_test_input_path(const ::std::string& value);
  inline void set_test_input_path(const char* value);
  inline void set_test_input_path(const char* value, size_t size);
  inline ::std::string* mutable_test_input_path();
  inline ::std::string* release_test_input_path();
  inline void set_allocated_test_input_path(::std::string* test_input_path);

  // optional string test_output_path = 15 [default = "./test_new_data"];
  inline bool has_test_output_path() const;
  inline void clear_test_output_path();
  static const int kTestOutputPathFieldNumber = 15;
  inline const ::std::string& test_output_path() const;
  inline void set_test_output_path(const ::std::string& value);
  inline void set_test_output_path(const char* value);
  inline void set_test_output_path(const char* value, size_t size);
  inline ::std::string* mutable_test_output_path();
  inline ::std::string* release_test_output_path();
  inline void set_allocated_test_output_path(::std::string* test_output_path);

  // optional float min_prob = 16 [default = 0.5];
  inline bool has_min_prob() const;
  inline void clear_min_prob();
  static const int kMinProbFieldNumber = 16;
  inline float min_prob() const;
  inline void set_min_prob(float value);

  // @@protoc_insertion_point(class_scope:rdf.RDFParameter)
 private:
  inline void set_has_num_images();
  inline void clear_has_num_images();
  inline void set_has_num_samples();
  inline void clear_has_num_samples();
  inline void set_has_num_trees();
  inline void clear_has_num_trees();
  inline void set_has_num_depth();
  inline void clear_has_num_depth();
  inline void set_has_max_span();
  inline void clear_has_max_span();
  inline void set_has_space_size();
  inline void clear_has_space_size();
  inline void set_has_num_features();
  inline void clear_has_num_features();
  inline void set_has_num_thresholds();
  inline void clear_has_num_thresholds();
  inline void set_has_input_path();
  inline void clear_has_input_path();
  inline void set_has_output_path();
  inline void clear_has_output_path();
  inline void set_has_out_name();
  inline void clear_has_out_name();
  inline void set_has_test_input_path();
  inline void clear_has_test_input_path();
  inline void set_has_test_output_path();
  inline void clear_has_test_output_path();
  inline void set_has_min_prob();
  inline void clear_has_min_prob();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::int32 num_images_;
  ::google::protobuf::int32 num_samples_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > num_pixels_;
  ::google::protobuf::RepeatedField<int> color_;
  ::google::protobuf::int32 num_trees_;
  ::google::protobuf::int32 num_depth_;
  ::google::protobuf::int32 max_span_;
  ::google::protobuf::int32 space_size_;
  ::google::protobuf::int32 num_features_;
  ::google::protobuf::int32 num_thresholds_;
  static ::std::string* _default_input_path_;
  ::std::string* input_path_;
  static ::std::string* _default_output_path_;
  ::std::string* output_path_;
  static ::std::string* _default_out_name_;
  ::std::string* out_name_;
  static ::std::string* _default_test_input_path_;
  ::std::string* test_input_path_;
  static ::std::string* _default_test_output_path_;
  ::std::string* test_output_path_;
  float min_prob_;
  friend void  protobuf_AddDesc_rdf_2eproto();
  friend void protobuf_AssignDesc_rdf_2eproto();
  friend void protobuf_ShutdownFile_rdf_2eproto();

  void InitAsDefaultInstance();
  static RDFParameter* default_instance_;
};
// ===================================================================


// ===================================================================

// RDFParameter

// optional int32 num_images = 1 [default = 10000];
inline bool RDFParameter::has_num_images() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void RDFParameter::set_has_num_images() {
  _has_bits_[0] |= 0x00000001u;
}
inline void RDFParameter::clear_has_num_images() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void RDFParameter::clear_num_images() {
  num_images_ = 10000;
  clear_has_num_images();
}
inline ::google::protobuf::int32 RDFParameter::num_images() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_images)
  return num_images_;
}
inline void RDFParameter::set_num_images(::google::protobuf::int32 value) {
  set_has_num_images();
  num_images_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_images)
}

// optional int32 num_samples = 2 [default = 2000];
inline bool RDFParameter::has_num_samples() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void RDFParameter::set_has_num_samples() {
  _has_bits_[0] |= 0x00000002u;
}
inline void RDFParameter::clear_has_num_samples() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void RDFParameter::clear_num_samples() {
  num_samples_ = 2000;
  clear_has_num_samples();
}
inline ::google::protobuf::int32 RDFParameter::num_samples() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_samples)
  return num_samples_;
}
inline void RDFParameter::set_num_samples(::google::protobuf::int32 value) {
  set_has_num_samples();
  num_samples_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_samples)
}

// repeated int32 num_pixels = 3;
inline int RDFParameter::num_pixels_size() const {
  return num_pixels_.size();
}
inline void RDFParameter::clear_num_pixels() {
  num_pixels_.Clear();
}
inline ::google::protobuf::int32 RDFParameter::num_pixels(int index) const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_pixels)
  return num_pixels_.Get(index);
}
inline void RDFParameter::set_num_pixels(int index, ::google::protobuf::int32 value) {
  num_pixels_.Set(index, value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_pixels)
}
inline void RDFParameter::add_num_pixels(::google::protobuf::int32 value) {
  num_pixels_.Add(value);
  // @@protoc_insertion_point(field_add:rdf.RDFParameter.num_pixels)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
RDFParameter::num_pixels() const {
  // @@protoc_insertion_point(field_list:rdf.RDFParameter.num_pixels)
  return num_pixels_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
RDFParameter::mutable_num_pixels() {
  // @@protoc_insertion_point(field_mutable_list:rdf.RDFParameter.num_pixels)
  return &num_pixels_;
}

// repeated .rdf.RDFParameter.Col color = 4;
inline int RDFParameter::color_size() const {
  return color_.size();
}
inline void RDFParameter::clear_color() {
  color_.Clear();
}
inline ::rdf::RDFParameter_Col RDFParameter::color(int index) const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.color)
  return static_cast< ::rdf::RDFParameter_Col >(color_.Get(index));
}
inline void RDFParameter::set_color(int index, ::rdf::RDFParameter_Col value) {
  assert(::rdf::RDFParameter_Col_IsValid(value));
  color_.Set(index, value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.color)
}
inline void RDFParameter::add_color(::rdf::RDFParameter_Col value) {
  assert(::rdf::RDFParameter_Col_IsValid(value));
  color_.Add(value);
  // @@protoc_insertion_point(field_add:rdf.RDFParameter.color)
}
inline const ::google::protobuf::RepeatedField<int>&
RDFParameter::color() const {
  // @@protoc_insertion_point(field_list:rdf.RDFParameter.color)
  return color_;
}
inline ::google::protobuf::RepeatedField<int>*
RDFParameter::mutable_color() {
  // @@protoc_insertion_point(field_mutable_list:rdf.RDFParameter.color)
  return &color_;
}

// optional int32 num_trees = 5 [default = 10];
inline bool RDFParameter::has_num_trees() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void RDFParameter::set_has_num_trees() {
  _has_bits_[0] |= 0x00000010u;
}
inline void RDFParameter::clear_has_num_trees() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void RDFParameter::clear_num_trees() {
  num_trees_ = 10;
  clear_has_num_trees();
}
inline ::google::protobuf::int32 RDFParameter::num_trees() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_trees)
  return num_trees_;
}
inline void RDFParameter::set_num_trees(::google::protobuf::int32 value) {
  set_has_num_trees();
  num_trees_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_trees)
}

// optional int32 num_depth = 6 [default = 20];
inline bool RDFParameter::has_num_depth() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void RDFParameter::set_has_num_depth() {
  _has_bits_[0] |= 0x00000020u;
}
inline void RDFParameter::clear_has_num_depth() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void RDFParameter::clear_num_depth() {
  num_depth_ = 20;
  clear_has_num_depth();
}
inline ::google::protobuf::int32 RDFParameter::num_depth() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_depth)
  return num_depth_;
}
inline void RDFParameter::set_num_depth(::google::protobuf::int32 value) {
  set_has_num_depth();
  num_depth_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_depth)
}

// optional int32 max_span = 7 [default = 200];
inline bool RDFParameter::has_max_span() const {
  return (_has_bits_[0] & 0x00000040u) != 0;
}
inline void RDFParameter::set_has_max_span() {
  _has_bits_[0] |= 0x00000040u;
}
inline void RDFParameter::clear_has_max_span() {
  _has_bits_[0] &= ~0x00000040u;
}
inline void RDFParameter::clear_max_span() {
  max_span_ = 200;
  clear_has_max_span();
}
inline ::google::protobuf::int32 RDFParameter::max_span() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.max_span)
  return max_span_;
}
inline void RDFParameter::set_max_span(::google::protobuf::int32 value) {
  set_has_max_span();
  max_span_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.max_span)
}

// optional int32 space_size = 8 [default = 400];
inline bool RDFParameter::has_space_size() const {
  return (_has_bits_[0] & 0x00000080u) != 0;
}
inline void RDFParameter::set_has_space_size() {
  _has_bits_[0] |= 0x00000080u;
}
inline void RDFParameter::clear_has_space_size() {
  _has_bits_[0] &= ~0x00000080u;
}
inline void RDFParameter::clear_space_size() {
  space_size_ = 400;
  clear_has_space_size();
}
inline ::google::protobuf::int32 RDFParameter::space_size() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.space_size)
  return space_size_;
}
inline void RDFParameter::set_space_size(::google::protobuf::int32 value) {
  set_has_space_size();
  space_size_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.space_size)
}

// optional int32 num_features = 9 [default = 2000];
inline bool RDFParameter::has_num_features() const {
  return (_has_bits_[0] & 0x00000100u) != 0;
}
inline void RDFParameter::set_has_num_features() {
  _has_bits_[0] |= 0x00000100u;
}
inline void RDFParameter::clear_has_num_features() {
  _has_bits_[0] &= ~0x00000100u;
}
inline void RDFParameter::clear_num_features() {
  num_features_ = 2000;
  clear_has_num_features();
}
inline ::google::protobuf::int32 RDFParameter::num_features() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_features)
  return num_features_;
}
inline void RDFParameter::set_num_features(::google::protobuf::int32 value) {
  set_has_num_features();
  num_features_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_features)
}

// optional int32 num_thresholds = 10 [default = 50];
inline bool RDFParameter::has_num_thresholds() const {
  return (_has_bits_[0] & 0x00000200u) != 0;
}
inline void RDFParameter::set_has_num_thresholds() {
  _has_bits_[0] |= 0x00000200u;
}
inline void RDFParameter::clear_has_num_thresholds() {
  _has_bits_[0] &= ~0x00000200u;
}
inline void RDFParameter::clear_num_thresholds() {
  num_thresholds_ = 50;
  clear_has_num_thresholds();
}
inline ::google::protobuf::int32 RDFParameter::num_thresholds() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.num_thresholds)
  return num_thresholds_;
}
inline void RDFParameter::set_num_thresholds(::google::protobuf::int32 value) {
  set_has_num_thresholds();
  num_thresholds_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.num_thresholds)
}

// optional string input_path = 11 [default = "./data"];
inline bool RDFParameter::has_input_path() const {
  return (_has_bits_[0] & 0x00000400u) != 0;
}
inline void RDFParameter::set_has_input_path() {
  _has_bits_[0] |= 0x00000400u;
}
inline void RDFParameter::clear_has_input_path() {
  _has_bits_[0] &= ~0x00000400u;
}
inline void RDFParameter::clear_input_path() {
  if (input_path_ != _default_input_path_) {
    input_path_->assign(*_default_input_path_);
  }
  clear_has_input_path();
}
inline const ::std::string& RDFParameter::input_path() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.input_path)
  return *input_path_;
}
inline void RDFParameter::set_input_path(const ::std::string& value) {
  set_has_input_path();
  if (input_path_ == _default_input_path_) {
    input_path_ = new ::std::string;
  }
  input_path_->assign(value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.input_path)
}
inline void RDFParameter::set_input_path(const char* value) {
  set_has_input_path();
  if (input_path_ == _default_input_path_) {
    input_path_ = new ::std::string;
  }
  input_path_->assign(value);
  // @@protoc_insertion_point(field_set_char:rdf.RDFParameter.input_path)
}
inline void RDFParameter::set_input_path(const char* value, size_t size) {
  set_has_input_path();
  if (input_path_ == _default_input_path_) {
    input_path_ = new ::std::string;
  }
  input_path_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:rdf.RDFParameter.input_path)
}
inline ::std::string* RDFParameter::mutable_input_path() {
  set_has_input_path();
  if (input_path_ == _default_input_path_) {
    input_path_ = new ::std::string(*_default_input_path_);
  }
  // @@protoc_insertion_point(field_mutable:rdf.RDFParameter.input_path)
  return input_path_;
}
inline ::std::string* RDFParameter::release_input_path() {
  clear_has_input_path();
  if (input_path_ == _default_input_path_) {
    return NULL;
  } else {
    ::std::string* temp = input_path_;
    input_path_ = const_cast< ::std::string*>(_default_input_path_);
    return temp;
  }
}
inline void RDFParameter::set_allocated_input_path(::std::string* input_path) {
  if (input_path_ != _default_input_path_) {
    delete input_path_;
  }
  if (input_path) {
    set_has_input_path();
    input_path_ = input_path;
  } else {
    clear_has_input_path();
    input_path_ = const_cast< ::std::string*>(_default_input_path_);
  }
  // @@protoc_insertion_point(field_set_allocated:rdf.RDFParameter.input_path)
}

// optional string output_path = 12 [default = "./new_data"];
inline bool RDFParameter::has_output_path() const {
  return (_has_bits_[0] & 0x00000800u) != 0;
}
inline void RDFParameter::set_has_output_path() {
  _has_bits_[0] |= 0x00000800u;
}
inline void RDFParameter::clear_has_output_path() {
  _has_bits_[0] &= ~0x00000800u;
}
inline void RDFParameter::clear_output_path() {
  if (output_path_ != _default_output_path_) {
    output_path_->assign(*_default_output_path_);
  }
  clear_has_output_path();
}
inline const ::std::string& RDFParameter::output_path() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.output_path)
  return *output_path_;
}
inline void RDFParameter::set_output_path(const ::std::string& value) {
  set_has_output_path();
  if (output_path_ == _default_output_path_) {
    output_path_ = new ::std::string;
  }
  output_path_->assign(value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.output_path)
}
inline void RDFParameter::set_output_path(const char* value) {
  set_has_output_path();
  if (output_path_ == _default_output_path_) {
    output_path_ = new ::std::string;
  }
  output_path_->assign(value);
  // @@protoc_insertion_point(field_set_char:rdf.RDFParameter.output_path)
}
inline void RDFParameter::set_output_path(const char* value, size_t size) {
  set_has_output_path();
  if (output_path_ == _default_output_path_) {
    output_path_ = new ::std::string;
  }
  output_path_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:rdf.RDFParameter.output_path)
}
inline ::std::string* RDFParameter::mutable_output_path() {
  set_has_output_path();
  if (output_path_ == _default_output_path_) {
    output_path_ = new ::std::string(*_default_output_path_);
  }
  // @@protoc_insertion_point(field_mutable:rdf.RDFParameter.output_path)
  return output_path_;
}
inline ::std::string* RDFParameter::release_output_path() {
  clear_has_output_path();
  if (output_path_ == _default_output_path_) {
    return NULL;
  } else {
    ::std::string* temp = output_path_;
    output_path_ = const_cast< ::std::string*>(_default_output_path_);
    return temp;
  }
}
inline void RDFParameter::set_allocated_output_path(::std::string* output_path) {
  if (output_path_ != _default_output_path_) {
    delete output_path_;
  }
  if (output_path) {
    set_has_output_path();
    output_path_ = output_path;
  } else {
    clear_has_output_path();
    output_path_ = const_cast< ::std::string*>(_default_output_path_);
  }
  // @@protoc_insertion_point(field_set_allocated:rdf.RDFParameter.output_path)
}

// optional string out_name = 13 [default = "./forest.txt"];
inline bool RDFParameter::has_out_name() const {
  return (_has_bits_[0] & 0x00001000u) != 0;
}
inline void RDFParameter::set_has_out_name() {
  _has_bits_[0] |= 0x00001000u;
}
inline void RDFParameter::clear_has_out_name() {
  _has_bits_[0] &= ~0x00001000u;
}
inline void RDFParameter::clear_out_name() {
  if (out_name_ != _default_out_name_) {
    out_name_->assign(*_default_out_name_);
  }
  clear_has_out_name();
}
inline const ::std::string& RDFParameter::out_name() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.out_name)
  return *out_name_;
}
inline void RDFParameter::set_out_name(const ::std::string& value) {
  set_has_out_name();
  if (out_name_ == _default_out_name_) {
    out_name_ = new ::std::string;
  }
  out_name_->assign(value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.out_name)
}
inline void RDFParameter::set_out_name(const char* value) {
  set_has_out_name();
  if (out_name_ == _default_out_name_) {
    out_name_ = new ::std::string;
  }
  out_name_->assign(value);
  // @@protoc_insertion_point(field_set_char:rdf.RDFParameter.out_name)
}
inline void RDFParameter::set_out_name(const char* value, size_t size) {
  set_has_out_name();
  if (out_name_ == _default_out_name_) {
    out_name_ = new ::std::string;
  }
  out_name_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:rdf.RDFParameter.out_name)
}
inline ::std::string* RDFParameter::mutable_out_name() {
  set_has_out_name();
  if (out_name_ == _default_out_name_) {
    out_name_ = new ::std::string(*_default_out_name_);
  }
  // @@protoc_insertion_point(field_mutable:rdf.RDFParameter.out_name)
  return out_name_;
}
inline ::std::string* RDFParameter::release_out_name() {
  clear_has_out_name();
  if (out_name_ == _default_out_name_) {
    return NULL;
  } else {
    ::std::string* temp = out_name_;
    out_name_ = const_cast< ::std::string*>(_default_out_name_);
    return temp;
  }
}
inline void RDFParameter::set_allocated_out_name(::std::string* out_name) {
  if (out_name_ != _default_out_name_) {
    delete out_name_;
  }
  if (out_name) {
    set_has_out_name();
    out_name_ = out_name;
  } else {
    clear_has_out_name();
    out_name_ = const_cast< ::std::string*>(_default_out_name_);
  }
  // @@protoc_insertion_point(field_set_allocated:rdf.RDFParameter.out_name)
}

// optional string test_input_path = 14 [default = "./test_data"];
inline bool RDFParameter::has_test_input_path() const {
  return (_has_bits_[0] & 0x00002000u) != 0;
}
inline void RDFParameter::set_has_test_input_path() {
  _has_bits_[0] |= 0x00002000u;
}
inline void RDFParameter::clear_has_test_input_path() {
  _has_bits_[0] &= ~0x00002000u;
}
inline void RDFParameter::clear_test_input_path() {
  if (test_input_path_ != _default_test_input_path_) {
    test_input_path_->assign(*_default_test_input_path_);
  }
  clear_has_test_input_path();
}
inline const ::std::string& RDFParameter::test_input_path() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.test_input_path)
  return *test_input_path_;
}
inline void RDFParameter::set_test_input_path(const ::std::string& value) {
  set_has_test_input_path();
  if (test_input_path_ == _default_test_input_path_) {
    test_input_path_ = new ::std::string;
  }
  test_input_path_->assign(value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.test_input_path)
}
inline void RDFParameter::set_test_input_path(const char* value) {
  set_has_test_input_path();
  if (test_input_path_ == _default_test_input_path_) {
    test_input_path_ = new ::std::string;
  }
  test_input_path_->assign(value);
  // @@protoc_insertion_point(field_set_char:rdf.RDFParameter.test_input_path)
}
inline void RDFParameter::set_test_input_path(const char* value, size_t size) {
  set_has_test_input_path();
  if (test_input_path_ == _default_test_input_path_) {
    test_input_path_ = new ::std::string;
  }
  test_input_path_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:rdf.RDFParameter.test_input_path)
}
inline ::std::string* RDFParameter::mutable_test_input_path() {
  set_has_test_input_path();
  if (test_input_path_ == _default_test_input_path_) {
    test_input_path_ = new ::std::string(*_default_test_input_path_);
  }
  // @@protoc_insertion_point(field_mutable:rdf.RDFParameter.test_input_path)
  return test_input_path_;
}
inline ::std::string* RDFParameter::release_test_input_path() {
  clear_has_test_input_path();
  if (test_input_path_ == _default_test_input_path_) {
    return NULL;
  } else {
    ::std::string* temp = test_input_path_;
    test_input_path_ = const_cast< ::std::string*>(_default_test_input_path_);
    return temp;
  }
}
inline void RDFParameter::set_allocated_test_input_path(::std::string* test_input_path) {
  if (test_input_path_ != _default_test_input_path_) {
    delete test_input_path_;
  }
  if (test_input_path) {
    set_has_test_input_path();
    test_input_path_ = test_input_path;
  } else {
    clear_has_test_input_path();
    test_input_path_ = const_cast< ::std::string*>(_default_test_input_path_);
  }
  // @@protoc_insertion_point(field_set_allocated:rdf.RDFParameter.test_input_path)
}

// optional string test_output_path = 15 [default = "./test_new_data"];
inline bool RDFParameter::has_test_output_path() const {
  return (_has_bits_[0] & 0x00004000u) != 0;
}
inline void RDFParameter::set_has_test_output_path() {
  _has_bits_[0] |= 0x00004000u;
}
inline void RDFParameter::clear_has_test_output_path() {
  _has_bits_[0] &= ~0x00004000u;
}
inline void RDFParameter::clear_test_output_path() {
  if (test_output_path_ != _default_test_output_path_) {
    test_output_path_->assign(*_default_test_output_path_);
  }
  clear_has_test_output_path();
}
inline const ::std::string& RDFParameter::test_output_path() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.test_output_path)
  return *test_output_path_;
}
inline void RDFParameter::set_test_output_path(const ::std::string& value) {
  set_has_test_output_path();
  if (test_output_path_ == _default_test_output_path_) {
    test_output_path_ = new ::std::string;
  }
  test_output_path_->assign(value);
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.test_output_path)
}
inline void RDFParameter::set_test_output_path(const char* value) {
  set_has_test_output_path();
  if (test_output_path_ == _default_test_output_path_) {
    test_output_path_ = new ::std::string;
  }
  test_output_path_->assign(value);
  // @@protoc_insertion_point(field_set_char:rdf.RDFParameter.test_output_path)
}
inline void RDFParameter::set_test_output_path(const char* value, size_t size) {
  set_has_test_output_path();
  if (test_output_path_ == _default_test_output_path_) {
    test_output_path_ = new ::std::string;
  }
  test_output_path_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:rdf.RDFParameter.test_output_path)
}
inline ::std::string* RDFParameter::mutable_test_output_path() {
  set_has_test_output_path();
  if (test_output_path_ == _default_test_output_path_) {
    test_output_path_ = new ::std::string(*_default_test_output_path_);
  }
  // @@protoc_insertion_point(field_mutable:rdf.RDFParameter.test_output_path)
  return test_output_path_;
}
inline ::std::string* RDFParameter::release_test_output_path() {
  clear_has_test_output_path();
  if (test_output_path_ == _default_test_output_path_) {
    return NULL;
  } else {
    ::std::string* temp = test_output_path_;
    test_output_path_ = const_cast< ::std::string*>(_default_test_output_path_);
    return temp;
  }
}
inline void RDFParameter::set_allocated_test_output_path(::std::string* test_output_path) {
  if (test_output_path_ != _default_test_output_path_) {
    delete test_output_path_;
  }
  if (test_output_path) {
    set_has_test_output_path();
    test_output_path_ = test_output_path;
  } else {
    clear_has_test_output_path();
    test_output_path_ = const_cast< ::std::string*>(_default_test_output_path_);
  }
  // @@protoc_insertion_point(field_set_allocated:rdf.RDFParameter.test_output_path)
}

// optional float min_prob = 16 [default = 0.5];
inline bool RDFParameter::has_min_prob() const {
  return (_has_bits_[0] & 0x00008000u) != 0;
}
inline void RDFParameter::set_has_min_prob() {
  _has_bits_[0] |= 0x00008000u;
}
inline void RDFParameter::clear_has_min_prob() {
  _has_bits_[0] &= ~0x00008000u;
}
inline void RDFParameter::clear_min_prob() {
  min_prob_ = 0.5f;
  clear_has_min_prob();
}
inline float RDFParameter::min_prob() const {
  // @@protoc_insertion_point(field_get:rdf.RDFParameter.min_prob)
  return min_prob_;
}
inline void RDFParameter::set_min_prob(float value) {
  set_has_min_prob();
  min_prob_ = value;
  // @@protoc_insertion_point(field_set:rdf.RDFParameter.min_prob)
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace rdf

#ifndef SWIG
namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::rdf::RDFParameter_Col> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::rdf::RDFParameter_Col>() {
  return ::rdf::RDFParameter_Col_descriptor();
}

}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_rdf_2eproto__INCLUDED
